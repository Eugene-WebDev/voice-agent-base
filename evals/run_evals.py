#!/usr/bin/env python3
"""Eval harness for the RAG persona: retrieval quality, grounded answers, refusal on unknowns.

    .venv/bin/python evals/run_evals.py                 # real run (OPENAI_API_KEY + ANTHROPIC_API_KEY)
    .venv/bin/python evals/run_evals.py --no-judge      # skip the LLM-as-judge step (cheaper)
    .venv/bin/python evals/run_evals.py --offline       # fake embedder + fake LLM: wiring check, no API cost

Checks per case:
  retrieval     expected SOP file is in the top_k retrieved chunks (or nothing required for out-of-scope)
  deterministic every must_include group matches; no must_not_include string appears
  judge         a separate model scores groundedness 1–5 against ONLY the retrieved text
Exit code 1 if any threshold fails → usable as a regression gate in CI before changing prompt/model.

Note: the production dispatcher lets the model CALL `retrieve_context` as a tool. This harness injects
the retrieved context directly, so it measures retrieval + grounded generation, not tool choice.
"""
import argparse, asyncio, hashlib, json, os, re, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

THRESHOLDS = {"retrieval_hit_rate": 0.9, "deterministic_pass_rate": 0.8, "judge_grounded_avg": 4.0}
JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_PROMPT = """You grade an assistant's answer for GROUNDEDNESS. Use ONLY the retrieved context below.
Score 1-5: 5 = every claim is supported by the context (or it correctly says the context lacks the answer);
3 = mostly supported, minor unsupported detail; 1 = invents facts not in the context.
Return JSON only: {{"grounded": <1-5>, "reason": "<one sentence>"}}

QUESTION: {q}
RETRIEVED CONTEXT:
{ctx}
ANSWER:
{a}"""


# ---------- offline fakes (wiring check only) ----------
class FakeEmbedder:
    """Bag-of-words hashing — deterministic, no network. Not a quality measure."""
    async def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 256
            for w in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
            out.append(v)
        return out

class FakeLLM:
    async def generate(self, messages, *, system=None, max_tokens=1024, temperature=0.0, **_):
        from core.interfaces import LLMResult
        text = messages[-1].content
        if "RETRIEVED CONTEXT" in text:
            return LLMResult(content='{"grounded": 5, "reason": "offline"}')
        return LLMResult(content="[offline] " + text[:400], usage={"input_tokens": 0, "output_tokens": 0})


def norm(s):
    return s.lower()

def check_deterministic(answer, case):
    a = norm(answer)
    missing = [g for g in case.get("must_include", []) if not any(norm(x) in a for x in g)]
    forbidden = [x for x in case.get("must_not_include", []) if norm(x) in a]
    return {"pass": not missing and not forbidden, "missing": missing, "forbidden": forbidden}

def first_json(text):
    s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e + 1]) if 0 <= s < e else {}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default=str(ROOT / "personas/sample-rag.yaml"))
    ap.add_argument("--cases", default=str(ROOT / "evals/cases.yaml"))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()

    persona = yaml.safe_load(open(a.persona, encoding="utf-8"))
    cases = yaml.safe_load(open(a.cases, encoding="utf-8"))
    rag = persona["rag"]; llm_cfg = persona["llm"]; k = int(rag.get("top_k", 3))

    from core.interfaces import LLMMessage
    from rag.inmemory_store import InMemoryRAGStore
    if not a.offline:
        from dotenv import load_dotenv
        for f in (ROOT / ".env", ROOT / ".dotenv"):
            if f.exists(): load_dotenv(f)
        from adapters.llm.claude import ClaudeLLM
        llm = ClaudeLLM(llm_cfg["model"], api_key=os.environ.get("ANTHROPIC_API_KEY"))
        judge = ClaudeLLM(JUDGE_MODEL, api_key=os.environ.get("ANTHROPIC_API_KEY"))
        store = InMemoryRAGStore(corpus_path=ROOT / rag["corpus_path"], chunk_size=rag["chunk_size"],
                                 chunk_overlap=rag["chunk_overlap"], embedding_model=rag["embedding_model"])
    else:
        llm = judge = FakeLLM()
        store = InMemoryRAGStore(corpus_path=ROOT / rag["corpus_path"], chunk_size=rag["chunk_size"],
                                 chunk_overlap=rag["chunk_overlap"], embedder=FakeEmbedder())

    rows = []
    for c in cases:
        t0 = time.perf_counter()
        chunks = await store.retrieve(c["question"], k=k)
        sources = [ch.source for ch in chunks]
        ctx = "\n\n".join(f"[{ch.source}]\n{ch.text}" for ch in chunks)
        exp = c.get("expected_source")
        hit = (exp in sources) if exp else None
        user = f"Kontekst z bazy wiedzy:\n{ctx}\n\nPytanie: {c['question']}"
        res = await llm.generate([LLMMessage(role="user", content=user)], system=persona["system_prompt"],
                                 max_tokens=int(llm_cfg.get("max_tokens", 1024)), temperature=0.0)
        latency = round(time.perf_counter() - t0, 2)
        det = check_deterministic(res.content, c)
        grounded, reason = None, None
        if not a.no_judge:
            jr = await judge.generate([LLMMessage(role="user", content=JUDGE_PROMPT.format(
                q=c["question"], ctx=ctx or "(nothing retrieved)", a=res.content))], max_tokens=200, temperature=0.0)
            j = first_json(jr.content); grounded, reason = j.get("grounded"), j.get("reason")
        rows.append({"id": c["id"], "retrieval_hit": hit, "sources": sources, "deterministic": det,
                     "grounded": grounded, "judge_reason": reason, "latency_s": latency,
                     "usage": getattr(res, "usage", {}), "answer": res.content})
        mark = "✓" if det["pass"] and hit is not False else "✗"
        print(f"{mark} {c['id']:28} hit={hit!s:5} det={det['pass']!s:5} grounded={grounded} {latency}s")

    in_scope = [r for r in rows if r["retrieval_hit"] is not None]
    summary = {
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in in_scope) / max(len(in_scope), 1), 3),
        "deterministic_pass_rate": round(sum(r["deterministic"]["pass"] for r in rows) / len(rows), 3),
        "judge_grounded_avg": (round(sum(r["grounded"] for r in rows if r["grounded"]) /
                               max(sum(1 for r in rows if r["grounded"]), 1), 2) if not a.no_judge else None),
        "p50_latency_s": sorted(r["latency_s"] for r in rows)[len(rows) // 2],
        "cases": len(rows), "mode": "offline" if a.offline else "live", "model": llm_cfg["model"],
    }
    failed = [m for m, th in THRESHOLDS.items() if summary.get(m) is not None and summary[m] < th]
    outdir = ROOT / "evals/results"; outdir.mkdir(exist_ok=True)
    out = outdir / f"{datetime.now():%Y%m%d-%H%M%S}-{summary['mode']}.json"
    json.dump({"summary": summary, "thresholds": THRESHOLDS, "failed": failed, "rows": rows},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n" + json.dumps(summary, indent=2))
    print(("PASS ✓" if not failed else f"FAIL: {failed}") + f"   → {out.relative_to(ROOT)}")
    if a.offline: print("(offline mode verifies wiring only — scores are meaningless)")
    sys.exit(1 if failed and not a.offline else 0)

if __name__ == "__main__":
    asyncio.run(main())
