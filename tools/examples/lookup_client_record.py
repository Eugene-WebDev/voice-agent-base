from __future__ import annotations

from tools.registry import tool

# Phase 2 mock data. Real Airtable adapter lands in Phase 4+.
_MOCK_RECORDS: dict[str, dict[str, str]] = {
    "adam-kowalski": {
        "name": "Adam Kowalski",
        "phone": "+48 600 000 000",
        "interest": "3-pokojowe mieszkanie, Mokotów, do 900k PLN",
        "last_viewing": "2026-05-20 — ul. Puławska 100",
    },
    "anna-nowak": {
        "name": "Anna Nowak",
        "phone": "+48 700 000 000",
        "interest": "2-pokojowe, Wola lub Śródmieście, do 700k PLN",
        "last_viewing": "2026-05-18 — ul. Górczewska 50",
    },
}


@tool()
def lookup_client_record(client_id: str) -> str:
    """Look up a client's CRM record by their kebab-case ID.

    Returns name, phone, current interests, and last viewing date if found.
    """
    record = _MOCK_RECORDS.get(client_id.lower())
    if not record:
        return f"No client found with id '{client_id}'. Available: {sorted(_MOCK_RECORDS)}"
    return "\n".join(f"{k}: {v}" for k, v in record.items())
