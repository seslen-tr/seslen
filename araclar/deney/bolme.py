# -*- coding: utf-8 -*-
"""
UZUN CÜMLEYİ BÖLMEK (03.11.2026) — makale sınavında takılma cümle uzadıkça artıyordu (0–7 sözcük %9, 16–23 %26,
24+ %38): model uzun cümlede sesleri kaydırıyor ("soru" → "sonru", "kanun" → "kanın"). Uzun cümle virgül / noktalı
virgül / iki noktadan kısa parçalara bölünüp ayrı okunursa, aralarına yalnız virgül duraklaması konursa düzelir mi?
Makalelerden bağımsız sınama: turkce-uslup'un eğitim derleminden (haber, deneme, anlatı…) rastgele uzun cümleler
(≥ EN_AZ sözcük, en az bir virgül), seslen-tts paketinin kendi yolu, 4 ses.

Ölçü: CTC + GOP okunuş denetimi (araclar/olcu/denetim.py; bütün cümlenin sesi, bütün cümlenin metniyle), UTMOS,
hece/sn. Çıktı: cikti/bolme/ (sonuc.json, dinle.html).

    python araclar/deney/bolme.py [--adet 40] [--parca 12] [--denetim]   (--denetim: + paketin özdenetimli yolu)
"""
from __future__ import annotations

import html
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK.parents[1] / "ses_projeleri" / "seslen-tts"), str(KOK / "araclar" / "olcu")]
CIKTI = KOK / "cikti" / "bolme"
#: Genel metin: turkce-uslup'un eğitim derlemi (haber, bilgi, deneme, anlatı, masal…; makalelerle ilgisiz)
KAYNAK = KOK.parents[1] / "ses_projeleri" / "turkce-uslup" / "veri" / "egitim.tsv"
SESLER = ["zeynep", "nehir", "hasan", "emre"]
EN_AZ = 16
VIRGUL_MS = 150


def main(adet: int, parca: int) -> None:
    import denetim
    import soundfile as sf
    from dogallik import puan
    from seslen_tts import ORAN, Seslen
    from seslen_tts.model import _UNLU
    from turkce_normalize import virgulden_bol
    from turkce_vurgu import normallestir
    s = Seslen(klasor=KOK / "cikti" / "acik" / "seslen-tts", denetim=False)
    oz = Seslen(klasor=KOK / "cikti" / "acik" / "seslen-tts") if "--denetim" in sys.argv else None
    kipler = ("butun", "bolunmus") + (("denetimli",) if oz is not None else ())
    from turkce_normalize import cumleleri_ayir
    satir = [c for x in KAYNAK.read_text(encoding="utf-8").splitlines() if x for c in cumleleri_ayir(x.split("\t")[0])]
    aday = [c for c in satir if EN_AZ <= len(c.split()) <= 40 and re.search(r"[,;:]", c)
            and len(virgulden_bol(c, parca)) > 1]
    random.Random(0).shuffle(aday)
    cumleler = aday[:adet]
    sonuc = []
    for i, c in enumerate(cumleler):
        ses = SESLER[i % len(SESLER)]
        stil = s.model.stil(ses)
        hedef = s.dogal_hiz(ses, "notr")
        ref = " ".join(re.findall(r"[a-zçğıöşüâîû]+", normallestir(c)))
        hece = sum(ch in _UNLU for ch in s.model.okunus(c))
        bolum = virgulden_bol(c, parca)
        dalga = {"butun": s.model.konus(c, stil, hedef_hece=hedef)}
        p = []
        for b in bolum:
            p += [s.model.konus(b, stil, hedef_hece=hedef), np.zeros(int(ORAN * VIRGUL_MS / 1000), np.float32)]
        dalga["bolunmus"] = np.concatenate(p[:-1])
        if oz is not None:                               # paketin kendi yolu: bölme + özdenetim (seslen_tts.denetim)
            dalga["denetimli"] = oz.tum(c, ses=ses, ton="notr")
        k = {"ses": ses, "cumle": c, "parcalar": bolum}
        (CIKTI / ses).mkdir(parents=True, exist_ok=True)
        for kip, w in dalga.items():
            yol = CIKTI / ses / f"{i:03d}_{kip}.wav"
            sf.write(yol, w, ORAN)
            d = denetim.denetle(w, ORAN, ref)
            k[kip] = {"takildi": bool(d["bayrak"]), "neden": d["neden"], "duyulan": d["duyulan"],
                      "utmos": round(puan(yol), 2), "hiz": round(hece / (len(w) / ORAN), 2)}
        sonuc.append(k)
        print(json.dumps({"i": i, "ses": ses} | {kip: k[kip]["takildi"] for kip in kipler}), flush=True)
    ozet = {kip: {"takilan": sum(k[kip]["takildi"] for k in sonuc), "cumle": len(sonuc),
                  "utmos": round(float(np.mean([k[kip]["utmos"] for k in sonuc])), 2),
                  "hiz": round(float(np.mean([k[kip]["hiz"] for k in sonuc])), 2)} for kip in kipler}
    print(json.dumps(ozet, ensure_ascii=False, indent=1))
    (CIKTI / "sonuc.json").write_text(json.dumps({"ozet": ozet, "satir": sonuc}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    sat = "".join(f"<tr><td>{k['ses']}<br>{html.escape(k['cumle'])}</td>" + "".join(
        f"<td><audio controls preload=none src='{k['ses']}/{i:03d}_{kip}.wav'></audio>"
        + (" <b>takıldı</b>" if k[kip]["takildi"] else "") + "</td>" for kip in kipler) + "</tr>"
        for i, k in enumerate(sonuc))
    (CIKTI / "dinle.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Uzun cümle</title><style>body{font:14px sans-serif;margin:16px}"
        "td{padding:4px;border-bottom:1px solid #ddd;vertical-align:top}audio{height:28px;width:220px}</style>"
        f"<h1>Uzun cümle: bütün ve virgülden bölünmüş</h1><p>{html.escape(json.dumps(ozet, ensure_ascii=False))}</p>"
        "<table><tr><th>cümle</th>" + "".join(f"<th>{kip}</th>" for kip in kipler) + "</tr>" + sat + "</table>", encoding="utf-8")


if __name__ == "__main__":
    a = sys.argv
    main(int(a[a.index("--adet") + 1]) if "--adet" in a else 40, int(a[a.index("--parca") + 1]) if "--parca" in a else 12)
