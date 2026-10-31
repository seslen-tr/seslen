# -*- coding: utf-8 -*-
"""
DOĞALLIK ÖLÇÜSÜ — UTMOS (otomatik MOS tahmini, 1–5) ile öğretmen ve öğrenci, ses ses.

    python araclar/olcu/dogallik.py ogretmen            # eğitim kayıtlarından ses başına ADET örnek → taban
    python araclar/olcu/dogallik.py dinleti/ses ogr-epoch_2nd_00000   # öğrenci dinletisi dosyaları

UTMOS İngilizce dinleyici puanlarıyla eğitildi: Türkçe vurguyu YARGILAYAMAZ, mutlak değeri
düşük çıkar. Hırıltı, cızırtı, robotluk gibi yapaylığı yakalar; KARŞILAŞTIRMA için kullanılır
(öğrenci − öğretmen farkı, tur tur gelişim). Sonuç cikti/dogallik.json'a eklenir.
"""
from __future__ import annotations

import json
import random
import statistics as ist
import sys
from pathlib import Path

import librosa
import torch

KOK = Path(__file__).resolve().parents[2]
ADET = 24
_model = None


def puan(yol: Path) -> float:
    global _model
    if _model is None:
        _model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    x, _ = librosa.load(yol, sr=16000)
    with torch.no_grad():
        return float(_model(torch.from_numpy(x)[None], 16000))


def ogretmen() -> dict[str, float]:
    ozet = json.loads((KOK / "veri" / "egitim" / "ozet.json").read_text(encoding="utf-8"))["duygu"]
    grup: dict[str, list[str]] = {}
    for anahtar in ozet:
        grup.setdefault(anahtar.split("/")[0], []).append(anahtar)
    rnd = random.Random(5)
    return {ses: ist.mean(puan(KOK / "veri" / "egitim" / "wavs" / f"{a}.flac")
                          for a in rnd.sample(liste, min(ADET, len(liste)))) for ses, liste in grup.items()}


def ogrenci(klasor: Path, onek: str) -> dict[str, float]:
    grup: dict[str, list[float]] = {}
    for f in sorted(klasor.glob(f"{onek}-*.wav")):
        ses = f.stem[len(onek) + 1:].split("-")[0]
        grup.setdefault(ses, []).append(puan(f))
    return {ses: ist.mean(v) for ses, v in grup.items()}


def main() -> int:
    sys.path.insert(0, str(KOK / "ogretmen"))
    from yukle import sureci_hazirla
    sureci_hazirla()
    if sys.argv[1] == "ogretmen":
        etiket, sonuc = "ogretmen", ogretmen()
    else:
        etiket, sonuc = sys.argv[2], ogrenci(KOK / sys.argv[1], sys.argv[2])
    dosya = KOK / "cikti" / "dogallik.json"
    kayit = json.loads(dosya.read_text(encoding="utf-8")) if dosya.exists() else {}
    kayit[etiket] = sonuc
    dosya.write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
    taban = kayit.get("ogretmen", {})
    for ses, p in sonuc.items():
        fark = f"  (öğretmen {taban[ses]:.2f}, fark {p - taban[ses]:+.2f})" if ses in taban and etiket != "ogretmen" else ""
        print(f"{ses:9s} {p:.2f}{fark}")
    print(f"ORTALAMA {ist.mean(sonuc.values()):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
