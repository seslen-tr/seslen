# -*- coding: utf-8 -*-
"""
SESLENDİRME SINAVI (16.10) — her yeni modelde bütün seslerin sabit bir cümle setini okuması ve otomatik denetim:
kullanıcı binlerce cümleyi dinleyemez; sınav sayıyla söyler, kulak yalnız en kötülere gider.

Cümleler: veri/sinav/elle.tsv (153 zor durum: sayı, ad, soru, emir, hitap, kısa ünlem, masal, ürün, zor sözcük,
duygu) + veri/sinav/cv.tsv (Common Voice 27.0, eğitimde geçmeyen 120 genel cümle). Seslendirme stüdyoyla aynı yoldan
(ogrenci/studyo.py: üslup, ton, telaffuz sözlüğü; özdenetim ve difüzyon kapalı, --ozdenetim ile ürün yolu); denetim araclar/olcu/denetim.py
(sözcük telaffuz puanı, fazladan ses).

Çıktı cikti/sinav/<model öneki>/: <ses>/<no>_<parça>.flac, sonuc.jsonl, rapor.json, dinle.html (her sesin en kötü
okumaları). --karsi ile önceki modelin raporu yan yana yazılır.

    python araclar/olcu/sinav.py cikti/ogrenci/t7-epoch_2nd_00000.pth [--sesler hasan,fatma] [--karsi t7-]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "ogretmen"), str(KOK / "araclar" / "olcu")]
SINAV = KOK / "veri" / "sinav"
SIRA = ["damla", "kuzey", "ayse", "serdar", "zeynep", "emre", "selin", "fatma", "hasan", "elif", "can", "ece",
        "pitircik", "koca", "firfir", "kara"]
KATEGORI = {"sy": "sayı/tarih", "ad": "ad", "so": "soru", "em": "emir/uyarı", "hi": "hitap", "ku": "kısa", "ma": "masal",
            "ur": "ürün", "zo": "zor sözcük", "du": "duygu", "cv": "genel"}


def cumleler() -> list[tuple[str, str]]:
    sonuc = []
    for ad in ("elle.tsv", "cv.tsv"):
        for s in (SINAV / ad).read_text(encoding="utf-8").splitlines():
            if s.strip() and not s.startswith("#"):
                k, c = s.split("\t", 1)
                sonuc.append((k, c.strip()))
    return sonuc


def calistir(agirlik: Path, sesler: list[str], ozdenetim: bool = False) -> Path:
    import soundfile as sf
    import denetim
    from model import ORAN
    from studyo import Ayar, Studyo
    from turkce_vurgu import normallestir
    st = Studyo(agirlik)
    hedef = KOK / "cikti" / "sinav" / (st.model.onek.rstrip("-") + ("-ozdenetim" if ozdenetim else ""))
    hedef.mkdir(parents=True, exist_ok=True)
    yol = hedef / f"sonuc_{'_'.join(sesler)}.jsonl" if len(sesler) < len(SIRA) else hedef / "sonuc.jsonl"
    bitti = {(r["ses"], r["no"]) for r in map(json.loads, yol.read_text(encoding="utf-8").splitlines())} if yol.exists() else set()
    liste = cumleler()
    with open(yol, "a", encoding="utf-8") as f:
        for ses in sesler:
            (hedef / ses).mkdir(exist_ok=True)
            for no, (kat, cumle) in enumerate(liste, 1):
                if (ses, no) in bitti:
                    continue
                for j, p in enumerate(st.seslendir(cumle, Ayar(ses=ses, ozdenetim=ozdenetim, difuzyon=ozdenetim, kayit=False)), 1):
                    ref = " ".join(re.findall(r"[a-zçğıöşüâîû]+", normallestir(st.telaffuz.uygula(p.cumle))))
                    d = denetim.denetle(p.ses, ORAN, ref)
                    sf.write(hedef / ses / f"{no:03d}_{j}.flac", p.ses, ORAN, subtype="PCM_16")
                    f.write(json.dumps({"ses": ses, "no": no, "parca": j, "kat": kat, "cumle": p.cumle, "ton": p.ton,
                                        "sn": round(len(p.ses) / ORAN, 2), **{k: d[k] for k in ("bayrak", "neden", "gop",
                                                                                                  "duyulan")}},
                                       ensure_ascii=False) + "\n")
                f.flush()
            print(f"{ses}: bitti", flush=True)
    return hedef


def _oku(hedef: Path) -> list[dict]:
    return [json.loads(x) for f in sorted(hedef.glob("sonuc*.jsonl")) for x in f.read_text(encoding="utf-8").splitlines() if x]


def rapor(hedef: Path, karsi: str | None = None) -> dict:
    r = _oku(hedef)
    ozet: dict = {"model": hedef.name, "sesler": {}, "kategori": {}}
    sesler = defaultdict(list)
    kat = defaultdict(list)
    for x in r:
        sesler[x["ses"]].append(x)
        kat[x["kat"]].append(x)
    for s in [s for s in SIRA if s in sesler]:
        v = sesler[s]
        ozet["sesler"][s] = {"parca": len(v), "takilan_yuzde": round(100 * np.mean([x["bayrak"] for x in v]), 1),
                             "gop_ort": round(float(np.mean([x["gop"] for x in v])), 2)}
    for k, v in kat.items():
        ozet["kategori"][KATEGORI.get(k, k)] = round(100 * np.mean([x["bayrak"] for x in v]), 1)
    ozet["toplam_takilan_yuzde"] = round(100 * np.mean([x["bayrak"] for x in r]), 1) if r else None
    (hedef / "rapor.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=1), encoding="utf-8")
    eski = None
    if karsi and (KOK / "cikti" / "sinav" / karsi.rstrip("-") / "rapor.json").exists():
        eski = json.loads((KOK / "cikti" / "sinav" / karsi.rstrip("-") / "rapor.json").read_text(encoding="utf-8"))
    print(f"{hedef.name}: takılan %{ozet['toplam_takilan_yuzde']}" + (f" (önceki {eski['model']}: %{eski['toplam_takilan_yuzde']})" if eski else ""))
    for s, v in ozet["sesler"].items():
        e = eski["sesler"].get(s) if eski else None
        print(f"  {s:9s} %{v['takilan_yuzde']:5.1f}" + (f"   önceki %{e['takilan_yuzde']:5.1f}" if e else ""))
    print("  türe göre: " + ", ".join(f"{k} %{v}" for k, v in sorted(ozet["kategori"].items(), key=lambda a: -a[1])))
    sayfa(hedef, r)
    return ozet


def sayfa(hedef: Path, r: list[dict], en_cok: int = 12) -> None:
    sesler = defaultdict(list)
    for x in r:
        if x["bayrak"]:
            sesler[x["ses"]].append(x)
    kart = []
    for s in [s for s in SIRA if s in sesler]:
        v = sorted(sesler[s], key=lambda x: x["gop"])[:en_cok]
        sat = "".join(f'<div class="s"><span>{html.escape(x["cumle"])}<small>{html.escape(", ".join(x["neden"]))} · duyulan: '
                      f'{html.escape(x["duyulan"])}</small></span><audio controls preload="none" src="{s}/{x["no"]:03d}_{x["parca"]}.flac">'
                      f'</audio></div>' for x in v)
        kart.append(f'<section class="kart"><h2>{s.capitalize()} <span>{len(sesler[s])} takılan</span></h2>{sat}</section>')
    (hedef / "dinle.html").write_text(f'''<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seslendirme Sınavı</title><style>:root{{--z:#f7f7f5;--k:#fff;--y:#1d1d1b;--s:#6b6b66;--c:#e3e3df}}
@media (prefers-color-scheme:dark){{:root{{--z:#161615;--k:#1f1f1d;--y:#ecece8;--s:#a3a39d;--c:#33332f}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--z);color:var(--y);font:15px/1.5 system-ui,sans-serif}}main{{max-width:960px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 6px}}.not{{color:var(--s)}}.kart{{background:var(--k);border:1px solid var(--c);border-radius:10px;padding:12px 16px;margin:0 0 14px}}
h2{{font-size:17px;margin:0 0 6px}}h2 span{{color:var(--s);font-weight:400;font-size:14px}}
.s{{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:10px;align-items:center;padding:5px 0;border-top:1px solid var(--c)}}
.s span{{font-size:14px}}.s small{{display:block;color:var(--s);font-size:12px}}audio{{width:100%;height:34px}}@media (max-width:640px){{.s{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Seslendirme sınavı: {hedef.name}</h1><p class="not">Her sesin otomatik denetime takılan en kötü {en_cok}
okuması. Bütün cümleleri dinlemeniz gerekmez; sayılar rapor.json'da.</p>{"".join(kart)}</main>
<script>document.addEventListener("play",e=>document.querySelectorAll("audio").forEach(a=>{{if(a!==e.target)a.pause()}}),true)</script></body></html>''',
                                       encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("agirlik")
    ap.add_argument("--sesler", default=",".join(SIRA))
    ap.add_argument("--karsi", default=None, help="önceki modelin öneki (t7-): raporda yan yana")
    ap.add_argument("--yalniz-rapor", action="store_true")
    ap.add_argument("--ozdenetim", action="store_true",
                    help="ürün yolu: stüdyo kendini dinleyip onarır (çıktı <önek>-ozdenetim/); varsayılan model yalın")
    a = ap.parse_args()
    sesler = [s for s in a.sesler.split(",") if s]
    if a.yalniz_rapor:
        from model import Ogrenci  # noqa: F401
        onek = Path(a.agirlik).name.split("-")[0]
        rapor(KOK / "cikti" / "sinav" / onek, a.karsi)
    else:
        rapor(calistir(Path(a.agirlik), sesler, a.ozdenetim), a.karsi)
