from __future__ import annotations

from tools.registry import tool


@tool()
def draft_client_email(client_name: str, subject: str, body: str) -> str:
    """Draft an email to a client. Does NOT send — prepares draft for review.

    Use when the user wants to communicate with a client in writing.
    Returns the formatted draft for the user to review or send manually.
    """
    return (
        f"Draft prepared for {client_name}:\n"
        f"Subject: {subject}\n\n"
        f"{body}\n\n"
        f"[NOT SENT — awaiting your review]"
    )
