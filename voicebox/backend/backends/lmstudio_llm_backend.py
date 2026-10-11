"""
LLM backend that talks to LM Studio over HTTP (OpenAI-compatible API).

Why, when `qwen_llm_backend` exists: that one loads weights INTO the Voicebox
process, onto the same card as the voice model. Every refine model takes 1-8 GB
away from TTS. LM Studio usually runs anyway and keeps its own model, so refining
costs Voicebox exactly zero memory.

REASONING STAYS ON — without it the small models make a mess of the transcript.
The price is measured: on Gemma E2B 397 reasoning tokens and 3.38 s versus 0 tokens
and 0.29 s without it.

⛔ PITFALL: reasoning comes out of THE SAME budget as the answer. With
`max_tokens=300` the model reasoned for 412 tokens and returned an EMPTY `content` —
no error, no warning, just nothing. `refinement.py` asks for 2048, so a short
dictation fits; a very long one may not.

⚠️ Reasoning can be turned off ONLY with `reasoning_effort: "none"`.
`chat_template_kwargs={"enable_thinking": false}` (the Qwen3 way) does NOT work
here — it passes without an error and reasoning stays, so it looks like it works
and it does not. Checked: `usage.completion_tokens_details.reasoning_tokens` = 397
despite the flag.

⚠️ From a container the host may be unreachable because of the firewall (with an
INPUT policy of DROP the packets vanish silently = a timeout, not "connection
refused"). With ufw:
    sudo ufw allow from 172.19.0.0/16 to any port 1234 proto tcp
"""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

from . import DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TEMPERATURE

logger = logging.getLogger(__name__)

# The container network's gateway, not 127.0.0.1 — inside a container the loopback
# is the container's own. Set `LMSTUDIO_BASE_URL` for your network.
DEFAULT_BASE_URL = "http://172.19.0.1:1234/v1"

# Limit for one answer. Refine returns text as long as its input, so the ceiling
# guards against a loop, it is not a budget.
_TIMEOUT_S = 120


def _cfg(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class LMStudioLLMBackend:
    """LLM refine through LM Studio. Zero weights in the Voicebox process."""

    def __init__(self, model_size: str = "lmstudio"):
        # `model_size` is part of the protocol and `refinement.py` reads
        # `backend.model_size` to record what refined the text. Here it carries the
        # LM Studio MODEL NAME — empty means "whatever is loaded there".
        self.model_size = model_size
        self._model_id: Optional[str] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ HTTP

    @property
    def _base(self) -> str:
        return _cfg("LMSTUDIO_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        key = _cfg("LMSTUDIO_API_KEY")
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def _request(self, path: str, data: Optional[dict] = None) -> dict:
        req = urllib.request.Request(
            self._base + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"LM Studio HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"LM Studio unreachable at {self._base} ({e.reason}). "
                "A timeout instead of a refusal = the firewall drops the packets, see the file header."
            ) from e

    # ------------------------------------------------------------- protocol

    def is_loaded(self) -> bool:
        return self._model_id is not None

    async def load_model(self, model_size: Optional[str] = None) -> None:
        """Loads no weights — only finds out what LM Studio has loaded."""
        if self._model_id is not None:
            return
        async with self._lock:
            if self._model_id is not None:
                return
            await asyncio.to_thread(self._pick_model)

    def _pick_model(self) -> None:
        d = self._request("/models")
        available = [m["id"] for m in d.get("data", [])]
        if not available:
            raise RuntimeError("LM Studio has no model loaded")
        self._model_id = available[0]
        logger.info("LM Studio refine: %s", self._model_id)

    def unload_model(self) -> None:
        """Only forgets the name. ⛔ Does NOT unload the model from LM Studio — it
        belongs to the user, who may be using it elsewhere right now."""
        self._model_id = None

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        model_size: Optional[str] = None,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        await self.load_model(model_size)
        return await asyncio.to_thread(
            self._generate, prompt, system, max_tokens, temperature, examples
        )

    def _generate(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        examples: Optional[list[tuple[str, str]]],
    ) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        for user_text, assistant_text in examples or []:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": prompt})

        d = self._request(
            "/chat/completions",
            {
                "model": self._model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return (d["choices"][0]["message"].get("content") or "").strip()
