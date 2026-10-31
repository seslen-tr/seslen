# -*- coding: utf-8 -*-
"""
KADRO SAYFASI — yeni seslerin adaylarını dinleyip seçmek için: dinleti/kadro.html

Girdi: cikti/adaylar/<aday>/olcum.json (bulut/kaggle/tasarim.py üretir,
kaggleIs.py tasarim-indir indirir). Her aday 4 kayıt: nötr (kimlik kaydı),
neşeli, uyarı, merak — ses ve duygu aralığı birlikte duyulur.
Ses dosyaları dinleti/ses/ altına kopyalanır (gitignore); sayfa çevrimdışı açılır.

    python ogretmen/kadroYap.py
"""
from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "ogretmen"))
from kadro import DENEME, KADRO  # noqa: E402

SES = KOK / "dinleti" / "ses"
TURLER = ["yetişkin", "genç", "yaşlı", "çocuk", "çizgi"]
DUYGU_AD = {"notr": "nötr", "neseli": "neşeli", "uyari": "uyarı", "merak": "merak", "heyecan": "heyecan", "ozur": "özür"}


def aday_kutusu(aday: str, tarif: str) -> str:
    e = html.escape
    yol = KOK / "cikti" / "adaylar" / aday / "olcum.json"
    if not yol.exists():
        return f"<div class=aday><h4>{e(aday)}</h4><p class=soluk>henüz üretilmedi</p></div>"
    j = json.loads(yol.read_text(encoding="utf-8"))
    satir = []
    for o in j["olcum"]:
        ad = f"kadro-{aday}-{o['dosya']}"
        SES.mkdir(parents=True, exist_ok=True)
        shutil.copy(yol.parent / o["dosya"], SES / ad)
        kotu = o["dogruluk"] < 0.9 or o["netlik"] < 18
        not_ = f"<div class=kotu>Whisper: {e(o['duyulan'])}</div>" if o["dogruluk"] < 0.9 else ""
        satir.append(f"<div class=kayit><span class='etiket{' kotu' if kotu else ''}'>{DUYGU_AD[o['duygu']]}</span>"
                     f"<audio controls preload=none src='ses/{e(ad)}'></audio>"
                     f"<span class=olcu>doğ {o['dogruluk']:.2f} · net {o['netlik']}</span>{not_}</div>")
    return f"<div class=aday><h4>{e(aday)}</h4><p class=tarif>{e(tarif)}</p>{''.join(satir)}</div>"


def main() -> int:
    e = html.escape
    bolum = []
    for tur in TURLER:
        sesler = [(ad, s) for ad, s in KADRO.items() if s["tur"] == tur and not s["secilen"]]
        if not sesler:
            continue
        bolum.append(f"<h2>{tur.capitalize()}</h2>")
        for ad, s in sesler:
            kutular = "".join(aday_kutusu(a, t) for a, t in s["adaylar"].items())
            bolum.append(f"<section><h3>{e(ad.capitalize())} <span class=soluk>— {e(s['aciklama'])}</span></h3>"
                         f"<div class=izgara>{kutular}</div></section>")
    cumleler = "".join(f"<li><b>{DUYGU_AD[d]}</b>: {e(c)}</li>" for d, c in
                       DENEME)
    (KOK / "dinleti" / "kadro.html").write_text(f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Kadro adayları</title><style>
:root {{ --zemin:#f4f5f8; --yuzey:#fff; --metin:#1b2230; --soluk:#5f6b7c; --cizgi:#dde2ea; --kotu:#c0392b; --vurgu:#2f6fdb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --zemin:#0f1216; --yuzey:#171c23; --metin:#e8edf2; --soluk:#8b98a8; --cizgi:#2a323d; --kotu:#ff6b66; --vurgu:#7aa7ff; }} }}
body {{ margin:0; background:var(--zemin); color:var(--metin); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:24px 16px 64px; }}
h2 {{ margin-top:36px; border-bottom:2px solid var(--vurgu); padding-bottom:4px; }}
.soluk, .tarif, .olcu {{ color:var(--soluk); }} .tarif {{ font-size:12px; font-style:italic; margin:2px 0 8px; }}
.izgara {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:12px; }}
.aday {{ background:var(--yuzey); border:1px solid var(--cizgi); border-radius:12px; padding:12px; }}
.aday h4 {{ margin:0; }} .kayit {{ display:grid; grid-template-columns:64px 1fr; gap:2px 8px; align-items:center; margin:6px 0; }}
.etiket {{ font-size:12px; font-weight:600; }} .olcu {{ grid-column:2; font-size:11px; }}
.kotu {{ color:var(--kotu); }} .kayit .kotu:not(.etiket) {{ grid-column:2; font-size:12px; }}
audio {{ width:100%; height:32px; }}
</style></head><body><main><h1>Kadro adayları</h1>
<p class=soluk>Her sesin 3 adayı var; her aday aynı 4 cümleyi okur. İlk kayıt sesin kimliğidir (tariften
tasarlandı), diğerleri o kimlik + duygu tarifi. Her ses için bir aday seç (ör. "ayse-2"); hiçbiri olmadıysa
"yeniden" de, tarifi değiştirip yeni aday üretirim. Kırmızı = ölçü kapısının altında (yanlış okuma ya da hırıltı).</p>
<ol>{cumleler}</ol>{''.join(bolum)}</main></body></html>""", encoding="utf-8")
    print("dinleti/kadro.html yazıldı")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
