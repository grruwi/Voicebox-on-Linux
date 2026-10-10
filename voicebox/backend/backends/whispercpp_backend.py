"""
STT przez whisper.cpp — `whisper-server` jako dziecko Voicebota (`services/whisper_serwer.py`).

Po co: `pytorch_backend` ciagnie whispera do procesu Voicebota w PyTorchu, na
te sama karte co glos. Tu model zyje w osobnym procesie ggml (HIP na ROCm 10,
ten sam backend co vox) albo na procesorze (`-ng`), wstaje przy zaladowaniu
i schodzi przy rozladowaniu. Do 30.09 byl to whisper z hosta na :8081.

⚠️ Backend Voicebota ma tez BLAD, ktory tedy znika: `WhisperProcessor` bez
`truncation=False` przycina wejscie do **30 sekund** i reszta przepada bez
sladu. whisper.cpp obsluguje dlugie nagrania sam.

Endpoint zgodny z OpenAI: POST /v1/audio/transcriptions (multipart).
"""

import asyncio
import json
import logging
import mimetypes
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from ..services import whisper_serwer

logger = logging.getLogger(__name__)

_TIMEOUT_S = 600


def _multipart(pola: dict, plik: Path) -> tuple[bytes, str]:
    """Sklejone recznie, zeby nie ciagnac `requests` do venv, ktory jest :ro."""
    granica = f"----voicebox{uuid.uuid4().hex}"
    czesci: list[bytes] = []
    for k, v in pola.items():
        czesci.append(
            f"--{granica}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        )
    typ = mimetypes.guess_type(plik.name)[0] or "application/octet-stream"
    czesci.append(
        f"--{granica}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{plik.name}\"\r\nContent-Type: {typ}\r\n\r\n".encode()
    )
    czesci.append(plik.read_bytes())
    czesci.append(f"\r\n--{granica}--\r\n".encode())
    return b"".join(czesci), f"multipart/form-data; boundary={granica}"


class WhisperCppSTTBackend:
    """STT bez wag w procesie Voicebota. Model zyje w `whisper-server`."""

    @property
    def model_size(self) -> str:
        """Model, ktory proces TRZYMA; gdy nic nie chodzi — ten z panelu.

        `check_model_loaded` / `unload_model_by_config` porownuja to pole
        z wpisem w rejestrze (medium/large/turbo), wiec musi mowic o procesie.
        """
        zal = whisper_serwer.zaladowany()
        if zal and zal[0]:
            return zal[0]
        return whisper_serwer.wybrane()[0]

    @property
    def wybrany_rozmiar(self) -> str:
        """Model z panelu — ten bierze `/transcribe`, gdy wolajacy nie poda `model`."""
        return whisper_serwer.wybrane()[0]

    def is_loaded(self) -> bool:
        return whisper_serwer.zaladowany() is not None

    # Warstwy wyzej pytaja o cache modelu, zeby pokazac pasek pobierania.
    # Pliki ggml leza w voice-models (:ro) — brak pliku zglasza `load_model`.
    def _is_model_cached(self, model_size: Optional[str] = None) -> bool:
        return True

    def _get_model_path(self, model_size: Optional[str] = None) -> str:
        model, kwant, _ = whisper_serwer.wybrane()
        return str(whisper_serwer.plik(model_size or model, kwant))

    async def load_model(self, model_size: Optional[str] = None) -> None:
        """Podnosi `whisper-server` z modelem; kwant i urzadzenie z panelu."""
        await asyncio.to_thread(whisper_serwer.uruchom, model_size)

    def unload_model(self) -> None:
        whisper_serwer.zatrzymaj()

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
        prompt: Optional[str] = None,
        carry_initial_prompt: Optional[bool] = None,
        max_len: Optional[int] = None,
    ) -> str:
        await self.load_model(model_size)
        return await asyncio.to_thread(
            self._transkrybuj, audio_path, language, prompt, carry_initial_prompt, max_len
        )

    def _transkrybuj(
        self,
        audio_path: str,
        language: Optional[str],
        prompt: Optional[str],
        carry_initial_prompt: Optional[bool],
        max_len: Optional[int],
    ) -> str:
        plik = Path(audio_path)
        if not plik.exists():
            raise RuntimeError(f"nie ma pliku: {audio_path}")

        pola = {"response_format": "json"}
        # ⛔ Bez jezyka whisper.cpp wykrywa sam z pierwszych 30 s (serwer stoi
        # na `-l auto`) i celowo go tu nie nadpisujemy, gdy wolajacy nic nie
        # poda — angielskie zdanie na sile po polsku wychodzi belkotem.
        if language:
            pola["language"] = language
        # Pola dyktafonu — puste nie ida wcale, serwer bierze swoje domyslne.
        if prompt:
            pola["prompt"] = prompt
        if carry_initial_prompt is not None:
            pola["carry_initial_prompt"] = "true" if carry_initial_prompt else "false"
        if max_len is not None:
            pola["max_len"] = str(max_len)

        dane, typ = _multipart(pola, plik)
        req = urllib.request.Request(whisper_serwer.URL, data=dane, headers={"Content-Type": typ})
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
                odp = json.loads(r.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"whisper-server nie odpowiada ({e})") from e

        return (odp.get("text") or "").strip()
