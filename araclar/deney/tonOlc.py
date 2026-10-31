# -*- coding: utf-8 -*-
"""
TON ÖLÇÜSÜ — öğrenci her duyguyu öğretmen kadar ayırabiliyor mu?

Her ses × duygu için öğretmen kayıtlarından ADET cümle seçilir; aynı metinler öğrenciyle o duygunun stil
vektörüyle seslendirilir. Akustik özellikler:
  perde (sesli çerçevelerde F0 ortalaması, yarım ton), dalga (F0 sapması, yarım ton), güç (RMS, dB),
  hız (saniyedeki harf), seslilik (sesli çerçeve oranı; fısıltıda düşer)
Her duygu için "nötrden fark" öğretmende ve öğrencide ayrı hesaplanır; oran = öğrenci farkı / öğretmen farkı
(özellikler boyunca, öğretmen farkının büyüklüğüyle ağırlıklı). 1'e yakın: duygu öğrenilmiş; 0'a yakın:
öğrenci duyguyu nötrden ayıramıyor. Sonuç: cikti/ton_olcu.json + ekrana tablo.

    python araclar/deney/tonOlc.py [ağırlık]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "ogretmen")]
SESLER = ["damla", "kuzey", "ayse", "serdar", "zeynep", "hasan"]
TONLAR = ["notr", "neseli", "heyecan", "uyari", "merak", "ozur", "uzgun", "kizgin", "fisilti"]
ADET = 6
OZ = ["perde", "dalga", "guc", "hiz", "seslilik"]


def ozellik(x: np.ndarray, sr: int, metin: str) -> np.ndarray:
    x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else x.astype(np.float32)
    x, _ = librosa.effects.trim(x, top_db=35)
    f0, sesli, _ = librosa.pyin(x, fmin=60, fmax=600, sr=16000, frame_length=1024, hop_length=256)
    f = f0[sesli & ~np.isnan(f0)]
    yt = 12 * np.log2(f / 100) if len(f) > 5 else np.array([np.nan])
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
    harf = sum(c.isalpha() for c in metin)
    return np.array([np.nanmean(yt), np.nanstd(yt), 20 * np.log10(np.mean(rms) + 1e-6),
                     harf / max(len(x) / 16000, 0.1), float(np.mean(sesli))])


def main() -> int:
    from yukle import sureci_hazirla
    from model import ORAN, Ogrenci
    sureci_hazirla()
    agirlik = sys.argv[1] if len(sys.argv) > 1 else str(KOK / "cikti" / "ogrenci" / "t10-epoch_2nd_00000.pth")
    o = Ogrenci(agirlik)
    ozet = json.loads((KOK / "veri" / "egitim" / "ozet.json").read_text(encoding="utf-8"))["duygu"]
    rnd = random.Random(7)
    sonuc: dict = {}
    for ses in SESLER:
        metin = {}
        for s in (KOK / "cikti" / "veri" / ses / "olcum.jsonl").read_text(encoding="utf-8").splitlines():
            j = json.loads(s)
            metin[j["kimlik"]] = j["metin"]
        ogr, ogt = {}, {}
        for ton in TONLAR:
            adaylar = [k.split("/")[1] for k, d in ozet.items() if k.startswith(ses + "/") and d == ton
                       and k.split("/")[1] in metin]
            if len(adaylar) < ADET:
                continue
            a, b = [], []
            for kimlik in rnd.sample(adaylar, ADET):
                x, sr = sf.read(KOK / "veri" / "egitim" / "wavs" / ses / f"{kimlik}.flac", dtype="float32")
                a.append(ozellik(x, sr, metin[kimlik]))
                w = o.konus(metin[kimlik], o.stil_dosyadan(ses, ton))
                b.append(ozellik(w, ORAN, metin[kimlik]))
            ogt[ton], ogr[ton] = np.nanmean(a, 0), np.nanmean(b, 0)
        sonuc[ses] = {}
        for ton in TONLAR[1:]:
            if ton not in ogt:
                continue
            dt, ds = ogt[ton] - ogt["notr"], ogr[ton] - ogr["notr"]
            # özellikleri ölçekle (yarım ton, dB, harf/sn, oran farklı birimler): öğretmenin tonlar arası yayılımıyla
            yay = np.nanstd(np.array([ogt[t] for t in ogt]), 0) + 1e-6
            dt_, ds_ = dt / yay, ds / yay
            oran = float(np.dot(ds_, dt_) / max(np.dot(dt_, dt_), 1e-9))
            sonuc[ses][ton] = {"oran": round(oran, 2), "ogretmen_fark": dict(zip(OZ, np.round(dt, 2).tolist())),
                               "ogrenci_fark": dict(zip(OZ, np.round(ds, 2).tolist()))}
        print(ses, " ".join(f"{t}:{v['oran']:.2f}" for t, v in sonuc[ses].items()), flush=True)
    ort = {t: round(float(np.mean([sonuc[s][t]["oran"] for s in sonuc if t in sonuc[s]])), 2) for t in TONLAR[1:]}
    print("\nORTALAMA (1 = öğretmen kadar ayırt edici, 0 = nötrden farksız):")
    for t, v in sorted(ort.items(), key=lambda x: x[1]):
        print(f"  {t:8s} {v:5.2f}")
    (KOK / "cikti" / "ton_olcu.json").write_text(json.dumps({"ortalama": ort, "sesler": sonuc}, ensure_ascii=False,
                                                            indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
