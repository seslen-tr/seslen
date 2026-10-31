# -*- coding: utf-8 -*-
"""
ZAYIF DUYGU TEŞHİSİ — bir ses × duygu nötrden neden ayrılmıyor: öğretmende mi yok (veri), öğrenci mi öğrenemedi?

Her hedef için öğretmen kayıtlarından N duygulu + N nötr cümle; öğrenci aynı metinleri okur. Özellikler (tonOlc'unkiler
+ merak için cümle sonu): perde ortancası, perde dalgalanması, güç, hız, SON YÜKSELİŞ (sesli çerçevelerin son %20'sinin
perdesi − tümünün ortancası, yarım ton; Türkçe soruda ezgi sonda yükselir, ortalama perde bunu kaçırır).
Etki büyüklüğü = (duygu − nötr) / ortak sapma; öğretmen ve öğrenci yan yana.

    python araclar/deney/zayifTeshis.py [ağırlık]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "ogretmen")]
HEDEF = [("damla", "uyari"), ("damla", "merak"), ("serdar", "merak")]
OZ = ["perde", "dalga", "guc", "hiz", "son"]
N = 25


def ozellik(x: np.ndarray, sr: int, metin: str) -> np.ndarray:
    import librosa
    x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else x.astype(np.float32)
    x, _ = librosa.effects.trim(x, top_db=35)
    f0, sesli, _ = librosa.pyin(x, fmin=55, fmax=500, sr=16000, frame_length=1024, hop_length=256)
    f = f0[sesli & ~np.isnan(f0)]
    if len(f) < 10:
        return np.full(5, np.nan)
    yt = 12 * np.log2(f / 100)
    son = np.median(yt[-max(3, len(yt) // 5):]) - np.median(yt)
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
    harf = sum(c.isalpha() for c in metin)
    return np.array([np.median(yt), np.std(yt), 20 * np.log10(np.mean(rms) + 1e-6), harf / max(len(x) / 16000, .1), son])


def etki(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.nanmean(a, 0) - np.nanmean(b, 0)) / (np.sqrt((np.nanvar(a, 0) + np.nanvar(b, 0)) / 2) + 1e-6)


def main() -> int:
    from yukle import sureci_hazirla
    from model import ORAN, Ogrenci
    sureci_hazirla()
    agirlik = sys.argv[1] if len(sys.argv) > 1 else str(KOK / "cikti" / "ogrenci" / "t10-epoch_2nd_00000.pth")
    o = Ogrenci(agirlik)
    ozet = json.loads((KOK / "veri" / "egitim" / "ozet.json").read_text(encoding="utf-8"))["duygu"]
    rnd = random.Random(4)
    sonuc = {}
    for ses, ton in HEDEF:
        metin = {}
        for s in (KOK / "cikti" / "veri" / ses / "olcum.jsonl").read_text(encoding="utf-8").splitlines():
            j = json.loads(s)
            metin[j["kimlik"]] = j["metin"]
        oz = {}
        for t in ("notr", ton):
            ad = [k.split("/")[1] for k, d in ozet.items() if k.startswith(ses + "/") and d == t and k.split("/")[1] in metin]
            ogt, ogr = [], []
            for k in rnd.sample(ad, N):
                x, sr = sf.read(KOK / "veri" / "egitim" / "wavs" / ses / f"{k}.flac", dtype="float32")
                ogt.append(ozellik(x, sr, metin[k]))
                ogr.append(ozellik(o.konus(metin[k], o.stil_dosyadan(ses, t)), ORAN, metin[k]))
            oz[t] = (np.array(ogt), np.array(ogr))
        e_ogt, e_ogr = etki(oz[ton][0], oz["notr"][0]), etki(oz[ton][1], oz["notr"][1])
        f_ogt = np.nanmean(oz[ton][0], 0) - np.nanmean(oz["notr"][0], 0)
        f_ogr = np.nanmean(oz[ton][1], 0) - np.nanmean(oz["notr"][1], 0)
        sonuc[f"{ses}/{ton}"] = {"ogretmen_etki": dict(zip(OZ, np.round(e_ogt, 2).tolist())),
                                 "ogrenci_etki": dict(zip(OZ, np.round(e_ogr, 2).tolist())),
                                 "ogretmen_fark": dict(zip(OZ, np.round(f_ogt, 2).tolist())),
                                 "ogrenci_fark": dict(zip(OZ, np.round(f_ogr, 2).tolist()))}
        print(f"\n{ses}/{ton}  (etki büyüklüğü; fark parantezde)")
        for i, ad in enumerate(OZ):
            print(f"  {ad:6s} öğretmen {e_ogt[i]:+.2f} ({f_ogt[i]:+.2f})   öğrenci {e_ogr[i]:+.2f} ({f_ogr[i]:+.2f})",
                  flush=True)
        print(f"  toplam öğretmen {np.linalg.norm(np.nan_to_num(e_ogt)):.2f}  öğrenci "
              f"{np.linalg.norm(np.nan_to_num(e_ogr)):.2f}", flush=True)
    (KOK / "cikti" / "zayif_teshis.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
