# -*- coding: utf-8 -*-
"""
UÇTAN UCA ZİNCİR — bir paragrafı bütün Türkçe katmanlardan geçirip seslendirir: dinleti/zincir.html

  1) cümlelere bölme        turkce-normalize (kısaltmada bölmez: "Dr.", "Cad.", "No.")
  2) ton + yoğunluk          turkce-duygu (konuşma eylemi kuralları + duygu modeli, varsa)
  3) metin hazırlama         turkce-normalize (tarih, saat, para, yüzde, kısaltma, adres…)
  4) okunuş (+ vurgu)        turkce-vurgu (3. aşama modeli vurgulu okunuşla eğitiliyor)
  5) ses                     öğrenci StyleTTS2, işlemcide; ton = stil vektörü, yoğunluk = nötrden uzaklık

    python araclar/deney/zincir.py [ağırlık.pth] [ses] ["metin"]
"""
from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "ogrenci"))
sys.path.insert(0, str(KOK / "ogretmen"))

METIN = ("Merhaba! Dr. Ayşe Yılmaz 15.03.2026 tarihinde saat 14:30'da İzmir'e gidecek. "
         "Bilet fiyatı %18 KDV dahil 1.250,50 TL. Acaba toplantı yine ertelenir mi? "
         "Dikkat, Atatürk Cad. No. 5'teki köprü onarımda, o yoldan geçmeyin! "
         "Kusura bakmayın, raporu geç gönderdik. Yaşasın, projemiz birinci oldu! "
         "Eski mahallemizin yıkıldığını duyunca içim burkuldu.")


def main() -> int:
    from turkce_duygu import Cozumleyici
    from turkce_normalize import normalize
    from turkce_vurgu.g2p import g2p
    from yukle import sureci_hazirla

    from model import ORAN, Ogrenci

    sureci_hazirla()
    agirlik = sys.argv[1] if len(sys.argv) > 1 else str(KOK / "cikti" / "ogrenci" / "t10-epoch_2nd_00000.pth")
    ses = sys.argv[2] if len(sys.argv) > 2 else "zeynep"
    metin = sys.argv[3] if len(sys.argv) > 3 else METIN
    o = Ogrenci(agirlik)
    v = np.load(KOK / "cikti" / "ogrenci" / "stiller.npz")
    stiller = {k: v[k] for k in v.files}
    mevcut = tuple(k.split("/")[1] for k in stiller if k.startswith(f"{ses}/"))
    coz = Cozumleyici(stiller=mevcut)
    print(f"ses {ses}, mevcut tonlar {mevcut}, duygu modeli {'var' if coz.model else 'yok (yalnız kurallar)'}")

    hedef = KOK / "dinleti" / "ses"
    hedef.mkdir(parents=True, exist_ok=True)
    notr = stiller[f"{ses}/notr"]
    satir, parcalar, toplam_is, toplam_ses = [], [], 0.0, 0.0
    for i, r in enumerate(coz.metin_cozumle(metin)):
        s = stiller[f"{ses}/{r.stil}"]
        if r.stil != "notr":                     # yoğunluk: nötrden tona doğru (0 → yarı, 1 → tam ton)
            s = notr + min(1.0, 0.5 + r.yogunluk) * (s - notr)
        t = time.perf_counter()
        w = o.konus(r.cumle, s)
        toplam_is += time.perf_counter() - t
        toplam_ses += len(w) / ORAN
        ad = f"zincir-{ses}-{i}.wav"
        sf.write(hedef / ad, w / max(1e-6, np.abs(w).max()) * 0.9, ORAN)
        parcalar += [w / max(1e-6, np.abs(w).max()) * 0.9, np.zeros(int(0.25 * ORAN), dtype=np.float32)]
        satir.append((r, normalize(r.cumle), g2p(r.cumle, vurgu=True), ad))
        print(f"{r.stil:8s} {r.yogunluk:.2f}  {r.cumle}\n         → {normalize(r.cumle)}\n         → {g2p(r.cumle, vurgu=True)}")
    sf.write(hedef / f"zincir-{ses}-tum.wav", np.concatenate(parcalar), ORAN)
    rtf = toplam_is / max(toplam_ses, 1e-6)
    e = html.escape
    govde = "".join(
        f"<tr><td>{e(r.cumle)}</td><td>{e(n)}</td><td><b>{e(r.stil)}</b> {r.yogunluk:.2f}"
        f"<div class=soluk>{e(r.eylem)} · {e(r.gerekce)}</div></td><td class=ipa>{e(ipa)}</td>"
        f"<td><audio controls preload=none src='ses/{e(ad)}'></audio></td></tr>" for r, n, ipa, ad in satir)
    (KOK / "dinleti" / "zincir.html").write_text(f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Uçtan uca zincir</title><style>
:root {{ --zemin:#f4f5f8; --yuzey:#fff; --metin:#1b2230; --soluk:#5f6b7c; --cizgi:#dde2ea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --zemin:#0f1216; --yuzey:#171c23; --metin:#e8edf2; --soluk:#8b98a8; --cizgi:#2a323d; }} }}
body {{ margin:0; background:var(--zemin); color:var(--metin); font:14px/1.45 system-ui, sans-serif; }}
main {{ max-width:1400px; margin:0 auto; padding:24px 16px 64px; }} p, .soluk {{ color:var(--soluk); }} .soluk {{ font-size:12px; }}
.kaydir {{ overflow-x:auto; }} table {{ border-collapse:collapse; background:var(--yuzey); border-radius:12px; width:100%; }}
th, td {{ border-bottom:1px solid var(--cizgi); padding:8px; text-align:left; vertical-align:top; }}
.ipa {{ font-family: "Noto Sans", "Segoe UI", sans-serif; }} audio {{ width:200px; height:30px; }}
</style></head><body><main><h1>Uçtan uca: metin → ton → okunuş → ses</h1>
<p>Ses: {e(ses)} · model: {e(Path(agirlik).name)} · işlemci RTF {rtf:.2f} (1 sn ses {rtf:.2f} sn'de) ·
duygu modeli: {'var' if coz.model else 'yok, yalnız kurallar'}. Vurgulu okunuş 3. aşama modeli içindir; bu model vurgusuz okunuşla eğitildi.</p>
<p><audio controls src='ses/zincir-{e(ses)}-tum.wav'></audio> paragrafın tamamı</p>
<div class=kaydir><table><tr><th>cümle</th><th>turkce-normalize</th><th>turkce-duygu</th><th>turkce-vurgu (vurgulu IPA)</th><th>ses</th></tr>
{govde}</table></div></main></body></html>""", encoding="utf-8")
    print(f"dinleti/zincir.html yazıldı; RTF {rtf:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
