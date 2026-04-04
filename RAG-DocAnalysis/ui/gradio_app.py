"""Gradio chat interface with conversation memory.

Two operating modes
-------------------
Direct mode  – imports the RAG chain directly (local dev, no API needed).
API mode     – forwards requests to the FastAPI server via HTTP.
               Activated when the  API_BASE_URL  environment variable is set.

Run standalone:
    python ui/gradio_app.py
Or via the CLI:
    python main.py ui
"""

from __future__ import annotations

import os
from typing import Any

import gradio as gr

from app.config import (
    DEFAULT_PRESET,
    DEFAULT_TEMPLATE,
    PARAMETER_PRESETS,
    PROMPT_TEMPLATES,
)

_API_BASE_URL: str = os.getenv("API_BASE_URL", "").rstrip("/")


# ---------------------------------------------------------------------------
# Query dispatcher (direct or via API)
# ---------------------------------------------------------------------------

def _query_direct(question: str, template: str, preset: str, chat_history: str) -> dict[str, Any]:
    from app.rag_chain import query
    return query(
        question=question,
        template_name=template,
        preset_name=preset,
        chat_history=chat_history,
    )


def _query_api(question: str, template: str, preset: str, chat_history: str) -> dict[str, Any]:
    import httpx
    payload = {
        "question": question,
        "template": template,
        "preset": preset,
        "chat_history": chat_history,
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{_API_BASE_URL}/ask", json=payload)
        resp.raise_for_status()
        return resp.json()


def run_query(question: str, template: str, preset: str, chat_history: str) -> dict[str, Any]:
    if _API_BASE_URL:
        return _query_api(question, template, preset, chat_history)
    return _query_direct(question, template, preset, chat_history)


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def _history_to_prompt_str(history: list[dict]) -> str:
    """Convert Gradio message-dict history to a plain-text prompt string."""
    lines = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core respond function
# ---------------------------------------------------------------------------

def respond(
    message: str,
    chat_history: list[dict],
    template_name: str,
    preset_name: str,
) -> tuple[list[dict], list[dict], str]:
    """
    Handle one user message.

    Returns
    -------
    (updated_chatbot_value, updated_state, sources_markdown)
    """
    if not message.strip():
        return chat_history, chat_history, ""

    history_str = _history_to_prompt_str(chat_history)

    try:
        result = run_query(
            question=message,
            template=template_name,
            preset=preset_name,
            chat_history=history_str,
        )
    except Exception as exc:  # noqa: BLE001
        error_msg = f"Error: {exc}"
        updated = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg},
        ]
        return updated, updated, ""

    answer: str = result["answer"]
    sources: list[dict] = result.get("sources", [])
    latency: float = result.get("latency_ms", 0.0)
    tokens: int | None = result.get("tokens_used")

    # Build sources panel markdown
    if sources:
        src_lines = "\n".join(
            f"- **{s['file']}** — chunk `{s['chunk_id']}`" for s in sources
        )
        sources_md = f"### Sources\n{src_lines}"
    else:
        sources_md = "### Sources\n*No sources retrieved.*"

    meta_parts = [f"Latency: **{latency:.0f} ms**"]
    if tokens is not None:
        meta_parts.append(f"Tokens: **{tokens}**")
    sources_md += f"\n\n*{' | '.join(meta_parts)}*"

    updated = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    return updated, updated, sources_md


def clear_conversation() -> tuple[list, list, str]:
    return [], [], ""


# ---------------------------------------------------------------------------
# Gradio Blocks UI
# ---------------------------------------------------------------------------

_TEMPLATE_DESCRIPTIONS = {
    name: name.capitalize() for name in PROMPT_TEMPLATES
}
_PRESET_DESCRIPTIONS = {
    name: f"{name.capitalize()} (temp={p.temperature}, top_p={p.top_p})"
    for name, p in PARAMETER_PRESETS.items()
}

with gr.Blocks(title="RAG Q&A Chatbot") as demo:

    gr.Markdown(
        "# RAG Q&A Chatbot\n"
        "Ask questions answered **only** from your document set. "
        "Every response cites the retrieved chunks it relied on.\n\n"
        f"**Mode:** {'API → ' + _API_BASE_URL if _API_BASE_URL else 'Direct (local RAG)'}"
    )

    with gr.Row():
        # ── Left: chat area ────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=520,
                buttons=["copy_all"],
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    label="Your question",
                    placeholder="Ask something about your documents…",
                    lines=2,
                    scale=5,
                    autofocus=True,
                )
                with gr.Column(scale=1, min_width=120):
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")

        # ── Right: settings + sources panel ───────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### Settings")
            template_dd = gr.Dropdown(
                choices=list(_TEMPLATE_DESCRIPTIONS.keys()),
                value=DEFAULT_TEMPLATE,
                label="Prompt Template",
                info="Controls tone and depth of the answer",
            )
            preset_dd = gr.Dropdown(
                choices=list(_PRESET_DESCRIPTIONS.keys()),
                value=DEFAULT_PRESET,
                label="LLM Preset",
                info="Temperature / Top-P settings",
            )

            gr.Markdown("---")
            gr.Markdown("### Retrieved Sources")
            sources_out = gr.Markdown(
                "*Sources will appear here after each answer.*",
            )

    # ── State: raw message history (mirrors chatbot) ───────────────────────
    history_state = gr.State([])

    # ── Wire up events ─────────────────────────────────────────────────────
    def _submit(message, history, template, preset):
        new_chatbot, new_state, sources_md = respond(message, history, template, preset)
        return new_chatbot, new_state, sources_md, ""   # clear the textbox

    send_btn.click(
        fn=_submit,
        inputs=[msg_box, history_state, template_dd, preset_dd],
        outputs=[chatbot, history_state, sources_out, msg_box],
    )
    msg_box.submit(
        fn=_submit,
        inputs=[msg_box, history_state, template_dd, preset_dd],
        outputs=[chatbot, history_state, sources_out, msg_box],
    )
    clear_btn.click(
        fn=clear_conversation,
        outputs=[chatbot, history_state, sources_out],
    )


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False,
    )
