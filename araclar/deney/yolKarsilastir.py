# -*- coding: utf-8 -*-
"""
YOL KARŞILAŞTIRMASI (14.10) — her ses için aynı 6 cümle (kadro.DENEME) üç hâliyle: dinleti sayfası
cikti/teshis/karsilastirma/dinle.html, sesin yanında "hangisi olsun" seçimi.
  şimdiki   t7 öğrenci (cikti/teshis/karsilastirma/simdiki/<ses>_<n>.wav)
  1. yol    iki aşamalı yapay ses, öğretmen çıktısı (cikti/teshis/tum_iki/<ses>_iki_<n>.wav; Hasan: hasan_iki/H_serdar_yasli)
  2. yol    Türkçe ön eğitimli öğrenci t8 (sesUzayi.py, git geçmişinde; 2. yol bırakıldı): yol2a/ aynı ses, yol2/ ses uzayından yeni
            "kimsesiz" ses (en yakın gerçek konuşmacıların ortalaması + bizim sesin karakteri)
1. yol öğretmen çıktısıdır: o ses seçilirse verisi yenilenip öğrenci yeniden eğitilir (fark ince ayarla kapanır).

    python araclar/deney/yolKarsilastir.py
"""
from __future__ import annotations

import html
import shutil
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "ogretmen"))
import kadro  # noqa: E402

T = KOK / "cikti" / "teshis"
H = T / "karsilastirma"
SIRA = ["damla", "kuzey", "ayse", "serdar", "zeynep", "emre", "selin", "fatma", "hasan", "elif", "can", "ece",
        "pitircik", "koca", "firfir", "kara"]
AD = {"ayse": "Ayşe", "pitircik": "Pıtırcık", "firfir": "Fırfır"}
# 19.10: seçimden sonra eğitilen t9 eklendi; 1. yol artık hedef (öğretmen), 2. yol aynı ses sütunu çıkarıldı
YOLLAR = [("simdiki", "Şimdiki (t7)"), ("t9", "t9"), ("t10", "Yeni model (t10)"), ("t10s", "t10 · stüdyo yolu"), ("yol1", "1. yol (hedef)")]


def yol1_kopyala() -> None:
    (H / "yol1").mkdir(parents=True, exist_ok=True)
    for ses in SIRA:
        for n in range(1, 7):
            k = {"hasan": T / "hasan_iki" / f"H_serdar_yasli_{n}.wav",               # Can: 15.10 seçilen erkek çocuk (A)
                 "can": T / "can_dene2" / f"can_p3_oglan_{n}.wav"}.get(ses, T / "tum_iki" / f"{ses}_iki_{n}.wav")
            if k.exists():
                shutil.copy(k, H / "yol1" / f"{ses}_{n}.wav")


def main() -> int:
    yol1_kopyala()
    cumle = [c for _, c in kadro.DENEME]
    var = {y: (H / y).exists() and any((H / y).glob("*.wav")) for y, _ in YOLLAR}
    yollar = [(y, ad) for y, ad in YOLLAR if var[y]]
    kartlar = []
    for ses in SIRA:
        bas = "".join(f"<th>{ad}</th>" for _, ad in yollar)
        satir = []
        for n, c in enumerate(cumle, 1):
            hucre = "".join(
                f'<td><audio controls preload="none" src="{y}/{ses}_{n}.wav"></audio></td>' if (H / y / f"{ses}_{n}.wav").exists()
                else "<td class='yok'>—</td>" for y, _ in yollar)
            satir.append(f"<tr><td class='c'>{n}. {html.escape(c)}</td>{hucre}</tr>")
        secim = "".join(f'<label><input type="radio" name="{ses}" value="{y}"> {ad}</label>' for y, ad in yollar)
        kartlar.append(f'''<section class="kart"><header><h2>{AD.get(ses, ses.capitalize())}</h2>
<span class="a">{html.escape(kadro.KADRO[ses]["aciklama"])}</span><span class="sec">{secim}</span></header>
<div class="tablo"><table><thead><tr><th>Cümle</th>{bas}</tr></thead><tbody>{"".join(satir)}</tbody></table></div></section>''')
    eksik = ""
    uzay = KOK / "cikti" / "teshis" / "uzay" / "rapor.json"
    if uzay.exists():
        import json
        r = json.loads(uzay.read_text(encoding="utf-8"))
        eksik += (f"<p class='not'>Uzay: {r.get('konusmaci', len(r['sesler']))} gerçek konuşmacı (" + ", ".join(f"{ {'yodas': 'YouTube', 'cv': 'Common Voice'}.get(k, k)} {n}" for k, n in r.get('kaynak', {}).items()) + "). "
                  f"Uzaydan yeni ses: en yakın gerçek konuşmacıya benzerlik "
                  f"{min(v['en_yakin_gercek'] for v in r['sesler'].values()):.2f}–{max(v['en_yakin_gercek'] for v in r['sesler'].values()):.2f}; "
                  f"gerçek konuşmacıların birbirine tipik en yakın benzerliği {r['tipik_en_yakin']:.2f}.</p>")
    sayfa = f'''<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ses Yolları Karşılaştırması</title><style>
:root{{--z:#f7f7f5;--k:#fff;--y:#1d1d1b;--s:#6b6b66;--c:#e3e3df;--v:#2f6f5e}}
@media (prefers-color-scheme:dark){{:root{{--z:#161615;--k:#1f1f1d;--y:#ecece8;--s:#a3a39d;--c:#33332f;--v:#6cc3a8}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--z);color:var(--y);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 90px}}h1{{font-size:22px;margin:0 0 6px}}.not{{color:var(--s);margin:0 0 16px}}
.kart{{background:var(--k);border:1px solid var(--c);border-radius:10px;padding:14px 16px;margin:0 0 14px}}
header{{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}}h2{{font-size:18px;margin:0}}.a{{color:var(--s);flex:1}}
.sec label{{margin-left:12px;font-weight:600;color:var(--v);cursor:pointer;white-space:nowrap}}
.tablo{{overflow-x:auto}}table{{width:100%;border-collapse:collapse}}th{{text-align:left;font-size:13px;color:var(--s);font-weight:600;padding:4px 6px}}
td{{border-top:1px solid var(--c);padding:4px 6px;vertical-align:middle}}td.c{{font-size:13.5px;min-width:220px}}td.yok{{color:var(--s);text-align:center}}
audio{{width:220px;height:34px}}
.alt{{position:fixed;bottom:0;left:0;right:0;background:var(--k);border-top:1px solid var(--c);padding:10px 16px;display:flex;gap:12px;align-items:center;justify-content:center;flex-wrap:wrap}}
button{{background:var(--v);color:#fff;border:0;border-radius:6px;padding:8px 14px;font-weight:600;cursor:pointer}}
</style></head><body><main><h1>Ses yolları karşılaştırması</h1>
<p class="not">Her ses aynı 6 cümleyi okuyor. Şimdiki: kullanılan model (t7). Yeni model (t9): seçimlerinize göre 1. yol verisiyle
(otomatik denetimden geçen kayıtlar) ve gerçek Türkçe konuşmayla eğitildi. 1. yol: öğretmenin okuması, t9'un yaklaşması gereken hedef.
2. yol: 9 saat gerçek Türkçe konuşmayla ön eğitimli model (pilot) —
aynı sesimiz ya da ses uzayından (YouTube + Common Voice gerçek konuşmacıları, sesin yaşı ve cinsiyetine uyanlar) yeni,
kimseye ait olmayan ses. Aksanı en iyi 5. ve 6. cümle gösterir.
Her sesin sağındaki seçimi işaretleyin, en alttan kopyalayın.</p>{eksik}{"".join(kartlar)}</main>
<div class="alt"><span id="say"></span><button id="kopya">Seçimleri kopyala</button></div>
<script>
const K="yol_secim",R=[...document.querySelectorAll("input[type=radio]")];let s={{}};
try{{s=JSON.parse(localStorage.getItem(K)||"{{}}")}}catch(e){{}}
R.forEach(r=>{{r.checked=s[r.name]===r.value;r.onchange=yaz}});
function secim(){{const o={{}};R.filter(r=>r.checked).forEach(r=>o[r.name]=r.value);return o}}
function yaz(){{const o=secim();document.getElementById("say").textContent=Object.keys(o).length+" / {len(SIRA)} ses seçildi";try{{localStorage.setItem(K,JSON.stringify(o))}}catch(e){{}}}}
document.getElementById("kopya").onclick=async()=>{{const o=secim(),ad={{simdiki:"şimdiki",yol1:"1. yol",yol2a:"2. yol aynı ses",yol2:"2. yol yeni ses"}};
const t="Seçimlerim: "+(Object.entries(o).map(([k,v])=>k+"="+ad[v]).join(", ")||"yok");
try{{await navigator.clipboard.writeText(t);document.getElementById("kopya").textContent="Kopyalandı"}}catch(e){{prompt("Kopyalayın:",t)}}}};
document.addEventListener("play",e=>document.querySelectorAll("audio").forEach(a=>{{if(a!==e.target)a.pause()}}),true);yaz();
</script></body></html>'''
    (H / "dinle.html").write_text(sayfa, encoding="utf-8")
    print(f"yazıldı: {H / 'dinle.html'} (sütunlar: {', '.join(ad for _, ad in yollar)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
