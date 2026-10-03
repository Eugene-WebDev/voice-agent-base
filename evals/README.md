# Evals — RAG persona regression suite

Measures whether the `sample-rag` persona finds the right SOP, answers from it, and **says it doesn't
know** instead of inventing numbers when the answer isn't in the corpus.

```bash
.venv/bin/python evals/run_evals.py             # live: OpenAI embeddings + persona model + judge
.venv/bin/python evals/run_evals.py --no-judge  # live, skip LLM-as-judge (cheaper)
.venv/bin/python evals/run_evals.py --offline   # no API calls — checks the harness wiring only
```

| Check | How | Threshold |
|---|---|---|
| Retrieval | expected SOP file is in the top-k chunks | hit rate ≥ 0.90 |
| Deterministic | required facts present (PL/EN alternatives), no forbidden strings (e.g. `%` on unknown questions) | pass rate ≥ 0.80 |
| Groundedness | a separate model (Claude Haiku) scores 1–5 against **only** the retrieved text | average ≥ 4.0 |

Exit code 1 when a threshold fails, so it can gate a prompt/model/chunking change in CI.
Results go to `evals/results/<timestamp>-<mode>.json` (git-ignored).

**Cases** (`cases.yaml`): 10 in-corpus questions across both SOPs + 2 out-of-scope questions
(commission, mortgage rate) that must be refused.

**Scope note:** production lets the model call `retrieve_context` as a tool; this harness injects the
retrieved context directly, so it tests retrieval + grounded generation, not the tool-choice decision.
A tool-choice suite is the natural next step.

Cost: 12 cases ≈ 12 embedding calls + 24 small model calls — cents per run.
