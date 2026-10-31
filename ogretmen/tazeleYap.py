# -*- coding: utf-8 -*-
"""
TAZELEME SAYFASI — dinleti/tazele.html: "daha kaliteli, Türkçeye yatkın" hangi kip?

Girdi: cikti/tazele/<aday>/olcum.json (bulut/kaggle/tazele.py). Her ses için:
  kimlik: eski ref.wav (tasarım, ~4 sn) ↔ yeni ref2.wav (tazelenmiş, ~10 sn)
  3 cümle × 3 kip:  A şimdiki (eski kimlik, adım 6) · B yeni kimlik, adım 10
                    · C yeni kimlik + devam kipi, adım 10
Üstte kip ortalamaları (Whisper doğruluğu, netlik). Karar kulakla.

    python ogretmen/tazeleYap.py
"""
from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "ogretmen"))
sys.path.insert(0, str(KOK / "araclar" / "olcu"))
from kadro import KADRO, KARSILASTIRMA  # noqa: E402
from olcu import benzerlik  # noqa: E402

SES = KOK / "dinleti" / "ses"
KIP = {"A": "şimdiki", "B": "yeni kimlik", "C": "yeni kimlik + devam"}


def kopya(kaynak: Path, ad: str) -> str:
    SES.mkdir(parents=True, exist_ok=True)
    shutil.copy(kaynak, SES / ad)
    return f"ses/{html.escape(ad)}"


def main() -> int:
    e = html.escape
    kayitlar = [json.loads(p.read_text(encoding="utf-8")) for p in (KOK / "cikti" / "tazele").glob("*/olcum.json")]
    sira = {s["secilen"]: i for i, s in enumerate(KADRO.values()) if s["secilen"]}
    kayitlar.sort(key=lambda k: (sira.get(k["aday"], 90), k["aday"]))
    for r in kayitlar:      # Kaggle eski olcu.py ile puanladı: güncel benzerlikle yeniden
        for x in r["karsilastirma"]:
            x["dogruluk"] = benzerlik(x["cumle"], x["duyulan"])
    ozet = []
    for k, ad in KIP.items():
        o = [x for r in kayitlar for x in r["karsilastirma"] if x["kip"] == k]
        if o:
            ozet.append(f"<tr><th>{k} — {ad}</th><td>{sum(x['dogruluk'] for x in o) / len(o):.3f}</td>"
                        f"<td>{sum(x['netlik'] for x in o) / len(o):.1f}</td>"
                        f"<td>{sum(x['dogruluk'] >= .9 and x['netlik'] >= 18 for x in o)}/{len(o)}</td></tr>")
    bolum = []
    for r in kayitlar:
        d = KOK / "cikti" / "tazele" / r["aday"]
        ad = r["aday"]
        t = r["tazele"]
        baslik = f"{r['ses'].capitalize()} ({ad})" if r["ses"] != ad else ad
        if r.get("yeni"):
            baslik += " — yeni tasarım"
        satir = [f"<tr><th>kimlik</th><td><audio controls preload=none src='{kopya(d / 'ref.wav', f'tz-{ad}-ref.wav')}'></audio>"
                 f"<div class=olcu>eski (tasarım)</div></td><td colspan=2><audio controls preload=none "
                 f"src='{kopya(d / 'ref2.wav', f'tz-{ad}-ref2.wav')}'></audio><div class=olcu>yeni, tazelenmiş · "
                 f"doğ {t['dogruluk']:.2f} · net {t['netlik']}</div></td></tr>"]
        for n, (duygu, cumle) in enumerate(KARSILASTIRMA):
            hucre = []
            for k in KIP:
                o = next(x for x in r["karsilastirma"] if x["kip"] == k and x["n"] == n)
                kotu = o["dogruluk"] < .9 or o["netlik"] < 18
                not_ = f"<div class=kotu>Whisper: {e(o['duyulan'])}</div>" if o["dogruluk"] < .9 else ""
                hucre.append(f"<td><audio controls preload=none src='{kopya(d / o['dosya'], f'tz-{ad}-{o['dosya']}')}'>"
                             f"</audio><div class='olcu{' kotu' if kotu else ''}'>doğ {o['dogruluk']:.2f} · net "
                             f"{o['netlik']}</div>{not_}</td>")
            satir.append(f"<tr><th>{e(cumle)}<div class=olcu>{duygu}</div></th>{''.join(hucre)}</tr>")
        bolum.append(f"<section><h3>{e(baslik)}</h3><table><tr><th></th>"
                     f"{''.join(f'<th>{k} — {v}</th>' for k, v in KIP.items())}</tr>{''.join(satir)}</table></section>")
    (KOK / "dinleti" / "tazele.html").write_text(f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Kimlik tazeleme</title><style>
:root {{ --zemin:#f4f5f8; --yuzey:#fff; --metin:#1b2230; --soluk:#5f6b7c; --cizgi:#dde2ea; --kotu:#c0392b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --zemin:#0f1216; --yuzey:#171c23; --metin:#e8edf2; --soluk:#8b98a8; --cizgi:#2a323d; --kotu:#ff6b66; }} }}
body {{ margin:0; background:var(--zemin); color:var(--metin); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:24px 16px 64px; }} p, .olcu {{ color:var(--soluk); }}
table {{ width:100%; border-collapse:collapse; background:var(--yuzey); border-radius:12px; overflow:hidden; margin-bottom:20px; }}
th, td {{ border-bottom:1px solid var(--cizgi); padding:8px; text-align:left; vertical-align:top; }}
th {{ width:230px; font-weight:600; }} .olcu {{ font-size:12px; font-weight:400; }} .kotu {{ color:var(--kotu); font-size:12px; }}
audio {{ width:100%; max-width:300px; height:32px; }}
@media (max-width:760px) {{ table, tr, td, th {{ display:block; width:auto; }} }}
</style></head><body><main><h1>Kimlik tazeleme — daha Türkçe, daha temiz mi?</h1>
<p>Her ses aynı 3 cümleyi üç kipte okur. <b>A</b> = şimdiki ayar. <b>B</b> = sesin ~10 sn'lik doğal Türkçe
bir paragrafla "tazelenmiş" kimliği. <b>C</b> = B + devam kipi (öğretmen önce o paragrafı okumuş gibi devam eder).
Her ses için A / B / C'den hangisi daha doğal, daha Türkçe? Serdar'da ayrıca kısıklığı gidermeye çalışan 3 yeni
tasarım var (serdar-5'in tarifinde "anadili Türkçe" ifadesi var).</p>
<p><b>Ölçü notu:</b> C kipinde duygulu cümlede tarif SESLİ okunuyor ("cheerful, happy…") ve cümle başına önceki paragrafın kuyruğu sızıyor ("Lö-öğleden") → C üretimde kullanılamaz. "Sapanka", "şemsiyenin" gibi kırmızıların çoğu Whisper'ın yazım alışkanlığı; kulakla karar ver.</p>
<table><tr><th>Kip</th><th>ort. doğruluk</th><th>ort. netlik</th><th>kapıdan geçen</th></tr>{''.join(ozet)}</table>
{''.join(bolum)}</main></body></html>""", encoding="utf-8")
    print(f"dinleti/tazele.html yazıldı ({len(kayitlar)} ses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
