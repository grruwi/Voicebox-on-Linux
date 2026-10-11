"""Strumien mowy do PRZEGLADARKI (11.10, decyzja grruwiego).

⭐ „Autoplay on generate" ZAZNACZONY = generacja z GUI nie idzie do `ffplay`, tylko tu:
`run_generation` (`_graj_strumieniem`) wrzuca kawalki z silnika w miare liczenia, a
`GET /generate/{id}/strumien` oddaje je przegladarce, ktora gra je od razu (Web Audio).
Odznaczony = stara droga, `ffplay` na glosniki serwera — tu nic nie trafia.

Format strumienia: 4 bajty czestotliwosci (uint32 LE), dalej surowe f32le mono.
Czestotliwosc w tresci, nie w naglowku HTTP — naglowek bez `expose_headers` jest
niewidoczny dla `fetch` z innego originu (panel z deva, Tauri).

Kawalki zostaja w pamieci do konca generacji + `_ZYCIE_PO_KONCU_S`, wiec przegladarka,
ktora podlaczy sie pozniej, dostaje wszystko od poczatku. Decyzja „nie strumieniem"
tez jest zapisywana (`odmow`), zeby trasa od razu oddala 204 i panel gral plik.

Wszystko w jednej petli zdarzen (generacja i trasa to korutyny) — bez blokad.
"""

import asyncio
import time
from typing import Optional

_ZYCIE_PO_KONCU_S = 60


class Strumien:
    def __init__(self, czestotliwosc: int):
        self.czestotliwosc = czestotliwosc
        self.kawalki: list[bytes] = []
        self.koniec = False
        self.blad = False
        self._zmiana = asyncio.Event()

    def dodaj(self, dane: bytes) -> None:
        self.kawalki.append(dane)
        self._budz()

    def zakoncz(self, blad: bool = False) -> None:
        self.koniec = True
        self.blad = blad
        self._budz()

    def _budz(self) -> None:
        # Czekajacy trzymaja STARY event — ustawiamy go i podstawiamy swiezy.
        self._zmiana.set()
        self._zmiana = asyncio.Event()

    async def czytaj(self):
        yield self.czestotliwosc.to_bytes(4, "little")
        i = 0
        while True:
            while i < len(self.kawalki):
                yield self.kawalki[i]
                i += 1
            if self.koniec:
                return
            await self._zmiana.wait()


# id generacji -> (czas ostatniej zmiany, Strumien albo None = „nie strumieniem")
_rejestr: dict[str, tuple[float, Optional[Strumien]]] = {}


def _sprzataj() -> None:
    teraz = time.monotonic()
    for gid, (czas, s) in list(_rejestr.items()):
        if (s is None or s.koniec) and teraz - czas > _ZYCIE_PO_KONCU_S:
            del _rejestr[gid]


def otworz(generation_id: str, czestotliwosc: int) -> Strumien:
    _sprzataj()
    s = Strumien(czestotliwosc)
    _rejestr[generation_id] = (time.monotonic(), s)
    return s


def zakoncz(generation_id: str, blad: bool = False) -> None:
    wpis = _rejestr.get(generation_id)
    if wpis and wpis[1] is not None:
        wpis[1].zakoncz(blad)
        _rejestr[generation_id] = (time.monotonic(), wpis[1])


def odmow(generation_id: str) -> None:
    """Ta generacja NIE leci do przegladarki — trasa odda 204, panel zagra plik."""
    _sprzataj()
    _rejestr[generation_id] = (time.monotonic(), None)


def pobierz(generation_id: str) -> tuple[bool, Optional[Strumien]]:
    """(czy decyzja juz zapadla, strumien albo None)."""
    wpis = _rejestr.get(generation_id)
    return (wpis is not None, wpis[1] if wpis else None)
