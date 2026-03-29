"""
Streaming wrapper around ollama.chat.
Broadcasts token chunks to the web dashboard log as the LLM generates,
so you can watch it think in real time.

Usage:
    from brain import llm_stream
    llm_stream.set_broadcast(my_log_fn)   # called once at startup
    raw = llm_stream.chat(messages, options, label="Scout")
"""
import time
from loguru import logger
import ollama
import config

_broadcast = None   # set by web/app.py at startup


def set_broadcast(fn):
    """Register the function used to push log lines to WebSocket clients."""
    global _broadcast
    _broadcast = fn


def chat(messages: list[dict], options: dict = None, label: str = "LLM") -> str:
    """
    Streaming ollama.chat call.
    Returns the full accumulated response string.
    Broadcasts partial output to the log every ~400ms so it's readable.
    """
    accumulated = ""
    last_broadcast = 0.0

    try:
        stream = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=messages,
            options=options or {},
            stream=True,
        )
        if _broadcast:
            _broadcast(f"[{label}] Generating response…")

        for chunk in stream:
            token = chunk["message"]["content"]
            accumulated += token
            now = time.time()
            if _broadcast and now - last_broadcast > 0.4:
                # Show the tail of the accumulated output — trim to one line
                preview = accumulated[-180:].replace("\n", " ").strip()
                _broadcast(f"[{label}] {preview}")
                last_broadcast = now

        if _broadcast:
            _broadcast(f"[{label}] ✓ Done ({len(accumulated)} chars)")

        return accumulated

    except Exception as e:
        logger.error(f"Ollama streaming failed: {e}")
        raise
