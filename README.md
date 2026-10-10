# Voicebox na audio.cpp (Vulkan)

[Voicebox](https://github.com/jamiepine/voicebox) (panel do klonowania głosu, MIT) przerobiony tak,
że mowa idzie z [audio.cpp](https://github.com/0xShug0/audio.cpp), a rozpoznawanie mowy
z [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Bez PyTorcha. Całość chodzi na Vulkanie,
więc wystarczy dowolna karta z Vulkanem.

Nieoficjalna przeróbka, niezwiązana z autorami Voiceboxa ani audio.cpp — błędy zgłaszaj tutaj, nie u nich.

Silniki mowy (oba klonują głos z próbki):
- **VoxCPM2**: streaming, pierwszy dźwięk po ~0,2 s; precyzja wag (q8/f16/f32) do wyboru w panelu;
- **OmniVoice** (k2-fsa): dyfuzja, generuje całą wypowiedź naraz, ~3× szybciej niż czas mowy;
  q8 albo f16, opcjonalny strumień kawałkami (po liniach albo zdaniach).

Trzy kontenery:

| kontener   | co robi                                                                 |
|------------|-------------------------------------------------------------------------|
| `voicebox` | API, panel webowy (port 17493), serwer MCP (`/mcp`), profile głosów      |
| `audiocpp` | silnik mowy z łatkami Voiceboxa; trzyma **jeden model naraz** (Vox albo OmniVoice), załadowanie jednego zdejmuje drugi |
| `whisper`  | whisper.cpp; model, kwant i karta/procesor wybiera się w panelu          |

## Wymagania

- Linux, Docker z `docker compose`
- karta z Vulkanem:
  - **AMD i Intel** działają od ręki (sterownik Mesa jest w obrazie, nowy, z trixie-backports);
  - **NVIDIA** jest niesprawdzona: potrzebny `nvidia-container-toolkit`, a w `docker-compose.yml` jest zakomentowany wariant;
- VRAM: Vox ~4–5 GB, OmniVoice ~3 GB, Whisper large-v3 q5 ~1,6 GB (Vox i OmniVoice nigdy naraz);
- dysk: ~7,4 GB na wagi, ~3 GB na obrazy (z wariantem ROCm ~6,3 GB).

## Start

```bash
git clone https://github.com/grruwi/voicebox-audiocpp.git
cd voicebox-audiocpp
./pobierz-modele.sh             # minimum ~7,4 GB; `./pobierz-modele.sh wszystko` = wszystkie warianty
cp .env.example .env            # opcjonalnie: serwer LLM do poprawiania transkrypcji
docker compose up -d --build    # pierwszy build kilka–kilkanaście min (kompilacja audio.cpp i whisper.cpp)
```

Panel: `http://<host>:17493`.

1. **Profile** → nowy profil → próbka głosu 5–30 s + jej dokładny transkrypt.
2. Silnik **VoxCPM2** albo **OmniVoice** → generuj.

Pierwsze ładowanie modelu trwa kilka sekund. Przy pierwszym starcie Vulkan kompiluje jeszcze shadery, więc zajmuje to dłużej; potem siedzą w `./dane/audiocpp-cache`.

Ustawienia → zakładki **Vox / OmniVoice / Whisper**: kwantyzacja, flash attention, kroki dyfuzji,
CFG, tempo mowy, guziki Przeładuj/Rozładuj.

## Uwagi

- **Backend** w zakładce Ogólne zostaw na **Vulkan**. Opcja ROCm działa tylko z wariantem ROCm (niżej).
- **Vox PyTorch** na liście silników potrzebuje osobnego kontenera z PyTorchem i ROCm. Tu go nie ma, nie wybieraj.
- **Port 17493 nie ma logowania.** Nie wystawiaj go do internetu bez reverse proxy z autoryzacją.
- `voicebox_speak` przez MCP gra na głośnikach **serwera**. Jeśli serwer ma dźwięk, odkomentuj w compose montaż socketu PulseAudio.
- Dane trzyma `./dane/voicebox`: baza, profile i wygenerowane pliki. Kopia zapasowa = kopia tego katalogu.
- Wagi VoxCPM2 to oryginał BF16. Silnik przepisuje go do wybranej precyzji przy każdym ładowaniu (~7 s z cache dysku).

## Opcjonalnie: ROCm (tylko Radeony)

**Niepotrzebne do działania** — Vulkan chodzi na każdej karcie i na RX 9070 XT był u nas szybszy.
Wariant ROCm dokłada do silnika binarkę HIP na ROCm 10 (TheRock), a Vulkan zostaje obok;
przełącza się w panelu (Ustawienia → Ogólne → Backend), bez przebudowy.

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d --build
```

Build pobiera ROCm (~2,4 GB na rodzinę kart, domyślnie RX 9000 + RX 7000) i kompiluje dłużej.
Inne karty: `ROCM_RODZINY` i `GPU_TARGETS` w `docker-compose.rocm.yml`.

## Co jest zmienione względem upstreamów

- `audiocpp/latki-voicebox.patch`: łatki na audio.cpp @ `497c7f0`:
  - stanowy dekoder strumienia VoxCPM2;
  - cache cech próbki głosu na dysku (zmiana profilu 2,4 s → kilka ms);
  - osobna precyzja części językowej;
  - prefill na flash attention (pierwszy dźwięk 270 → 170 ms).
- `audiocpp/zarzadca.py`, `whisper/zarzadca.py`: małe serwery HTTP, przez które Voicebox podnosi i zdejmuje procesy silników.
- `voicebox/`: fork Voiceboxa:
  - backend gada z audio.cpp i whisper.cpp po HTTP;
  - panel ma zakładki silników;
  - lista silników jest okrojona do VoxCPM2 i OmniVoice.

## Licencje

- `voicebox/` — Voicebox, [MIT](voicebox/LICENSE), © Voicebox Contributors; nasze zmiany na tej samej licencji.
- audio.cpp — [Apache 2.0](audiocpp/LICENSE-audio.cpp), © ShugoAI LLC. Kod pobiera się przy buildzie;
  `audiocpp/latki-voicebox.patch` to nasze zmiany w ich plikach (Apache 2.0).
- whisper.cpp — MIT, pobierany przy buildzie.
- ROCm (tylko wariant ROCm) — AMD, licencje w paczce, pobierany przy buildzie.

## Autorzy

Złożył [grruwi](https://github.com/grruwi) z Claude
