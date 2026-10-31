# -*- coding: utf-8 -*-
"""
SÖZCÜK GÜVENİ — Whisper'ın her sözcükten ne kadar emin olduğu (0-1).

Neden: doğruluk ölçüsü Whisper'ın YAZDIĞINA bakar; Whisper bildiği sözcüğü
bozuk okunsa da tamamlar (Pıtırcık "üniversite"yi yamuk okuyor, Whisper yine
"Üniversite" yazıyor; kullanıcı kulakla yakaladı, 01.10.2026). Sözcük
olasılığı düşükse okunuş belirsizdir → eğitimden çıkarılabilir.

SONUÇ (Pıtırcık, 1.400 kayıt): "Üniversite" sözcüğü 0,90-0,94 güvenle geçti —
Whisper bu bozukluktan da emin; en düşük güvenler çoğunlukla "bir", "ve", "bu"
gibi kısa sözcükler. Yani bu ölçü o okuma hatasını YAKALAMIYOR. Yalnız en kötü
%2 (< 0,3) elenir; bütün seslere 9 saat harcamaya değmedi. Asıl çare: öğrenci
16 sesle ortak eğitiliyor, telaffuzu bütün seslerden öğreniyor.

Her geçen kayda olcum.jsonl'de "min_guven" (en düşük sözcük olasılığı) ve
"min_sozcuk" eklenir; kaldığı yerden sürer (alanı olan kayıt atlanır).

    python araclar/olcu/kelimeGuven.py pitircik [ses …]     (boş = hepsi)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "araclar" / "olcu"))
sys.path.insert(0, str(KOK / "ogretmen"))
import olcu  # noqa: E402
from yukle import sureci_hazirla  # noqa: E402


def dosya(ses: str, kimlik: str) -> Path:
    for u in (".flac", ".wav"):
        p = KOK / "cikti" / "veri" / ses / "wavs" / f"{kimlik}{u}"
        if p.exists():
            return p
    raise FileNotFoundError(kimlik)


def guven(x: np.ndarray, sr: int) -> tuple[float, str]:
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        import soxr
        x = soxr.resample(x.astype(np.float32), sr, 16000)
    parca, _ = olcu._whisper().transcribe(x.astype(np.float32), language="tr", beam_size=1, word_timestamps=True)
    sozcukler = [w for p in parca for w in (p.words or [])]
    if not sozcukler:
        return 0.0, ""
    en = min(sozcukler, key=lambda w: w.probability)
    return round(float(en.probability), 3), en.word.strip()


def isle(ses: str) -> None:
    yol = KOK / "cikti" / "veri" / ses / "olcum.jsonl"
    kayit = [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines() if s.strip()]
    t0, n = time.time(), 0
    for i, o in enumerate(kayit):
        if not o["gecti"] or "min_guven" in o:
            continue
        x, sr = sf.read(dosya(ses, o["kimlik"]), dtype="float32")
        o["min_guven"], o["min_sozcuk"] = guven(x, sr)
        n += 1
        if n % 100 == 0:        # ara kayıt: kesilirse kaldığı yerden
            yol.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kayit), encoding="utf-8")
            print(f"{ses}: {n} ({(time.time() - t0) / n:.2f} sn/kayıt)", flush=True)
    yol.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kayit), encoding="utf-8")
    g = [o["min_guven"] for o in kayit if "min_guven" in o]
    print(f"{ses}: bitti, {len(g)} kayıt; min güven ortancası {np.median(g):.2f}, "
          f"<0,5: {sum(v < 0.5 for v in g)}, <0,3: {sum(v < 0.3 for v in g)}", flush=True)


if __name__ == "__main__":
    sureci_hazirla()        # EcoQoS kapalı: arka plandaki süreç güç tasarrufuna alınıp ~3× yavaşlıyordu
    for s in sys.argv[1:] or sorted(p.parent.name for p in (KOK / "cikti" / "veri").glob("*/olcum.jsonl")):
        isle(s)
