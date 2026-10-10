"""
LLM backend gadajacy po HTTP z LM Studio (API zgodne z OpenAI).

Po co, skoro jest `qwen_llm_backend`: tamten laduje wagi DO TEGO SAMEGO
procesu, czyli na te sama karte, na ktorej stoi VoxCPM. Kazdy model refine
to 1-8 GB zabrane TTS-owi. LM Studio i tak chodzi na kombajnie i trzyma
swoj model — wiec refine kosztuje Voicebox dokladnie zero pamieci.

ROZUMOWANIE ZOSTAJE WLACZONE — decyzja grruwiego 16.08: "to jest durny
model, bez reasoningu bedzie do podtarcia sie". Cena jest zmierzona: na E2B
397 tokenow myslenia i 3,38 s wobec 0 tokenow i 0,29 s bez niego.

⛔ PULAPKA: rozumowanie idzie z TEGO SAMEGO budzetu co odpowiedz. Przy
`max_tokens=300` model przemyslal 412 tokenow i zwrocil PUSTY `content` —
bez bledu, bez ostrzezenia, po prostu nic. `refinement.py` prosi o 2048,
wiec krotka dyktafonowa wypowiedz sie miesci, ale dluga juz niekoniecznie.
Stad `_generuj` powtarza zapytanie bez rozumowania, gdy wynik jest pusty:
lepiej gorszy tekst niz zaden.

⚠️ Wylaczyc rozumowanie mozna WYLACZNIE przez `reasoning_effort: "none"`.
`chat_template_kwargs={"enable_thinking": false}` (sposob z Qwen3) tutaj
NIE DZIALA — przechodzi bez bledu i rozumowanie zostaje, wiec wyglada na
dzialajace, a nie jest. Sprawdzone: `usage.completion_tokens_details.
reasoning_tokens` = 397 mimo flagi.

⚠️ Kontener NIE dosiega hosta bez reguly w ufw (polityka INPUT to DROP,
pakiety gina po cichu = timeout, nie "connection refused"):
    sudo ufw allow from 172.19.0.0/16 to any port 1234 proto tcp
"""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

from . import LLMBackend, DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TEMPERATURE

logger = logging.getLogger(__name__)

# Brama sieci kontenera, nie 127.0.0.1 — z wnetrza kontenera petla lokalna
# to jego wlasna petla, nie hosta.
DEFAULT_BASE_URL = "http://172.19.0.1:1234/v1"

# Limit na jedna odpowiedz. Refine zwraca tekst tej samej dlugosci co wejscie,
# wiec sufit jest zabezpieczeniem przed zapetleniem, nie budzetem.
_TIMEOUT_S = 120


def _cfg(nazwa: str, domyslnie: str = "") -> str:
    return os.environ.get(nazwa, domyslnie).strip()


class LMStudioLLMBackend:
    """LLM refine przez LM Studio. Zero wag w procesie Voicebota."""

    def __init__(self, model_size: str = "lmstudio"):
        # `model_size` jest w protokole i `refinement.py` czyta `backend.model_size`,
        # zeby zapisac w bazie, czym poprawiono. Tutaj niesie NAZWE MODELU w LM
        # Studio — pusta znaczy "cokolwiek jest tam zaladowane".
        self.model_size = model_size
        self._model_id: Optional[str] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ HTTP

    @property
    def _base(self) -> str:
        return _cfg("LMSTUDIO_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

    def _naglowki(self) -> dict:
        h = {"Content-Type": "application/json"}
        klucz = _cfg("LMSTUDIO_API_KEY")
        if klucz:
            h["Authorization"] = f"Bearer {klucz}"
        return h

    def _zapytaj(self, sciezka: str, dane: Optional[dict] = None) -> dict:
        req = urllib.request.Request(
            self._base + sciezka,
            data=json.dumps(dane).encode() if dane is not None else None,
            headers=self._naglowki(),
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            tresc = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"LM Studio HTTP {e.code}: {tresc}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"LM Studio nieosiagalne pod {self._base} ({e.reason}). "
                "Timeout zamiast odmowy = ufw zjada pakiety, patrz naglowek pliku."
            ) from e

    # ------------------------------------------------------------- protokol

    def is_loaded(self) -> bool:
        return self._model_id is not None

    async def load_model(self, model_size: Optional[str] = None) -> None:
        """Nie laduje wag — tylko ustala, co LM Studio ma zaladowane."""
        if self._model_id is not None:
            return
        async with self._lock:
            if self._model_id is not None:
                return
            await asyncio.to_thread(self._ustal_model)

    def _ustal_model(self) -> None:
        d = self._zapytaj("/models")
        dostepne = [m["id"] for m in d.get("data", [])]
        if not dostepne:
            raise RuntimeError("LM Studio nie ma zaladowanego zadnego modelu")
        self._model_id = dostepne[0]
        logger.info("LM Studio refine: %s", self._model_id)

    def unload_model(self) -> None:
        """Zapomina tylko nazwe. ⛔ NIE zdejmuje modelu z LM Studio — ten
        nalezy do grruwiego i moze go w tej chwili uzywac gdzie indziej."""
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
            self._generuj, prompt, system, max_tokens, temperature, examples
        )

    def _generuj(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        examples: Optional[list[tuple[str, str]]],
    ) -> str:
        wiadomosci: list[dict] = []
        if system:
            wiadomosci.append({"role": "system", "content": system})
        for user_text, assistant_text in examples or []:
            wiadomosci.append({"role": "user", "content": user_text})
            wiadomosci.append({"role": "assistant", "content": assistant_text})
        wiadomosci.append({"role": "user", "content": prompt})

        d = self._zapytaj(
            "/chat/completions",
            {
                "model": self._model_id,
                "messages": wiadomosci,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return (d["choices"][0]["message"].get("content") or "").strip()
