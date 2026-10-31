# -*- coding: utf-8 -*-
"""
SORU EZGİSİ ÖLÇÜSÜ — model.SORU_TEPE kuralı açık / kapalı, evet/hayır soruları (merak tonu):
  tepe   soru ekinden önceki bölgede (cümlenin %55-90'ı) en yüksek perde − ilk %50'nin ortancası (yarım ton)
  anlaşılırlık (Whisper), doğallık (UTMOS). Kulak için dinleti/soru.html.

    python araclar/deney/soruOlc.py
"""
from __future__ import annotations

import html
import statistics as ist
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "ogretmen"), str(KOK / "araclar" / "olcu"), str(KOK / "araclar" / "deney")]
SESLER = ["damla", "serdar", "kuzey", "zeynep", "hasan", "ayse"]
SORULAR = ["Bana yardımcı olabilir misin?", "Yarın bize gelecek misiniz?", "Bunu sen mi yaptın?",
           "Toplantı saat üçte mi başlıyor?", "Kapıyı kilitledin mi?", "Ankara'ya trenle mi gideceksiniz?",
           "Raporu bitirdiniz mi?", "Bu akşam sinemaya gidelim mi?", "Çay ister misiniz?",
           "Haritadaki parseller güncel mi?", "Annen de geliyor mu?", "Sesimi duyabiliyor musunuz?"]


def tepe(x: np.ndarray, sr: int) -> float:
    import librosa
    x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=16000)
    x, _ = librosa.effects.trim(x, top_db=35)
    f0, v, _ = librosa.pyin(x, fmin=55, fmax=500, sr=16000, frame_length=1024, hop_length=256)
    yt = 12 * np.log2(f0 / 100)
    n = len(yt)
    return float(np.nanmax(yt[int(n * .55):int(n * .9)]) - np.nanmedian(yt[:n // 2]))


def main() -> int:
    global ACIK
    import model as M
    from dogallik import puan
    from olcu import dogruluk
    from studyo import SES_AGIRLIK
    from turkce_normalize import normalize
    from yukle import sureci_hazirla
    sureci_hazirla()
    ACIK = M.SORU_TEPE
    modeller = {}
    for ses in SESLER:
        ad = SES_AGIRLIK.get(ses, "t10-epoch_2nd_00000.pth")
        modeller.setdefault(ad, M.Ogrenci(KOK / "cikti" / "ogrenci" / ad))
    hedef = KOK / "dinleti" / "ses"
    hedef.mkdir(parents=True, exist_ok=True)
    sonuc, satir = {}, []
    for ses in SESLER:
        o = modeller[SES_AGIRLIK.get(ses, "t10-epoch_2nd_00000.pth")]
        stil = o.stil_dosyadan(ses, "merak")
        hucre = {}
        for kip, tepe_yt in (("kapali", 0.0), ("acik", ACIK)):
            M.SORU_TEPE = tepe_yt
            t, d, u = [], [], []
            for i, c in enumerate(SORULAR):
                w = o.konus(normalize(c), stil)
                w = w / max(1e-6, np.abs(w).max()) * 0.9
                yol = hedef / f"soru-{ses}-{i:02d}-{kip}.wav"
                sf.write(yol, w, M.ORAN)
                t.append(tepe(w, M.ORAN))
                d.append(dogruluk(w, M.ORAN, normalize(c))[0])
                u.append(puan(yol))
                hucre.setdefault(i, {})[kip] = yol.name
            sonuc[(ses, kip)] = (ist.mean(t), ist.mean(d), min(d), ist.mean(u))
            print(f"{ses:7s} {kip:6s} tepe {ist.mean(t):+.2f} yt  anlaşılırlık {ist.mean(d):.3f} (en düşük {min(d):.2f})  "
                  f"UTMOS {ist.mean(u):.2f}", flush=True)
        for i, c in enumerate(SORULAR):
            satir.append(f"<tr><th>{ses.capitalize() if i == 0 else ''}</th><td>{html.escape(c)}</td>"
                         + "".join(f"<td><audio controls preload=none src='ses/{hucre[i][k]}'></audio></td>"
                                   for k in ("kapali", "acik")) + "</tr>")
    (KOK / "dinleti" / "soru.html").write_text(f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Soru ezgisi</title><style>
:root {{ --zemin:#f4f5f8; --yuzey:#fff; --metin:#1b2230; --soluk:#5f6b7c; --cizgi:#dde2ea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --zemin:#0f1216; --yuzey:#171c23; --metin:#e8edf2; --soluk:#8b98a8; --cizgi:#2a323d; }} }}
body {{ margin:0; background:var(--zemin); color:var(--metin); font:14px/1.45 system-ui, sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px; }} p {{ color:var(--soluk); }}
.kaydir {{ overflow-x:auto; }} table {{ border-collapse:collapse; background:var(--yuzey); }}
th, td {{ border-bottom:1px solid var(--cizgi); padding:6px 8px; text-align:left; }} audio {{ width:220px; height:30px; }}
</style></head><body><main><h1>Soru ezgisi: kural kapalı ↔ açık</h1>
<p>Evet/hayır sorularında ezginin tepesi soru ekinden önceki sözcüğün vurgulu hecesindedir ("yardımcı olaBİLİR misin?").
Sol: model kendi başına, sağ: kural açık (o heceye +{ACIK:g} yarım ton).</p>
<div class=kaydir><table><tr><th>ses</th><th>soru</th><th>kural kapalı</th><th>kural açık</th></tr>{''.join(satir)}</table></div>
</main></body></html>""", encoding="utf-8")
    print("dinleti/soru.html yazıldı")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
