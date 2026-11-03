# -*- coding: utf-8 -*-
"""
PERDE ÇEŞİTLİLİĞİ (03.11) — duzey.py'nin bulgusu: öğretmen her cümleyi başka bir perde düzeyinde okuyor (cümleler arası
düzey sapması 3,4 yarım ton), öğrenci hepsini aşağı yukarı aynı düzeyde (1,9). Öğrencinin perde tahmincisi öğretmen
perdesine L1 ile eğitildi: koşullu ortalamayı öğrenir, rastlantısal çeşitliliği öğrenemez. Çeşitliliği verebilecek
parça cümleye özgü difüzyon stili; tonlama yarısının ne kadarı ondan gelsin (model.DIF_BETA, şimdi 0,4)?
doku.py'nin 30 cümlesinde β ∈ {0,4; 0,7; 1,0}.

Ölçü: öğretmene benzerlik (WavLM-SV), MCD, UTMOS, takılan (CTC), cümleler arası düzey sapması (ses içinde), cümle içi
perde sapması, düzeyin öğretmenle ilişkisi (ses ortalaması çıkarılmış; r > 0 ise cümleye özgü düzey metinden
kestirilebiliyor).

    python araclar/deney/cesitlilik.py [--beta 0.4,0.7,1.0]     → cikti/doku/cesitlilik_*.json, cesitlilik.html
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "araclar" / "olcu"), str(KOK / "araclar" / "deney")]
CIKTI = KOK / "cikti" / "doku"
ORAN = 24000


def perde_ozeti(ogr, x: np.ndarray) -> tuple[float, float]:
    """(ortanca düzey, cümle içi sapma), yarım ton."""
    from ezgi import perde
    f = perde(ogr, x)
    st = 12 * np.log2(f[f > 40])
    return (float(np.median(st)), float(np.std(st))) if len(st) else (np.nan, np.nan)


def toplu(satir: list[dict], kip: str) -> dict:
    sesler = sorted({s["ses"] for s in satir})
    duz = {ses: np.array([[s[kip]["duzey"], s["ogretmen"]["duzey"]] for s in satir if s["ses"] == ses]) for ses in sesler}
    sapma = float(np.mean([np.std(d[:, 0]) for d in duz.values()]))
    a = np.concatenate([d[:, 0] - d[:, 0].mean() for d in duz.values()])
    b = np.concatenate([d[:, 1] - d[:, 1].mean() for d in duz.values()])
    v = [s[kip] for s in satir]
    o = {"duzey_sapmasi": round(sapma, 2), "duzey_r": round(float(np.corrcoef(a, b)[0, 1]), 3),
         "ic_sapma": round(float(np.mean([x["ic"] for x in v])), 2)}
    if kip != "ogretmen":
        o |= {a: round(float(np.mean([x[a] for x in v])), 3) for a in ("benzerlik", "mcd", "utmos")}
        o["takilan"] = sum(x["takildi"] for x in v)
    return o


def main(betalar: list[float]) -> None:
    import denetim
    import soundfile as sf
    from doku import Benzerlik, kepstrum, mcd
    from dogallik import puan
    from model import Ogrenci
    from prozodi import ipa_metin
    from studyo import URUN_AGIRLIK
    ogr = Ogrenci(KOK / "cikti" / "ogrenci" / URUN_AGIRLIK)
    sinama = json.loads((CIKTI / "sonuc.json").read_text(encoding="utf-8"))
    sv, satir = Benzerlik(), []
    for k in sinama:
        ses, ton, okunus, ad = k["ses"], k["ton"], k["okunus"], Path(k["dosya"]).stem
        ref_stil = ogr.stil_dosyadan(ses, ton if f"{ses}/{ton}" in ogr.stiller else "notr")
        x, _ = sf.read(CIKTI / ses / f"{ad}_ogretmen.wav", dtype="float32")
        g, kp = sv.gomme(x), kepstrum(ogr, x)
        d0, i0 = perde_ozeti(ogr, x)
        s = {"ses": ses, "ton": ton, "dosya": k["dosya"], "okunus": okunus, "ogretmen": {"duzey": d0, "ic": i0}}
        for b in betalar:
            kip = f"b{b:g}"
            w = ogr.konus("", ogr.cumle_stili(okunus, ref_stil, beta=b, okunus=okunus), okunus=okunus)
            p = CIKTI / ses / f"{ad}_{kip}.wav"
            sf.write(p, w, ORAN)
            d, i = perde_ozeti(ogr, w)
            s[kip] = {"duzey": d, "ic": i, "benzerlik": round(float(sv.gomme(w) @ g), 3), "utmos": round(puan(p), 2),
                      "mcd": round(mcd(kepstrum(ogr, w), kp), 2),
                      "takildi": bool(denetim.denetle(w, ORAN, ipa_metin(okunus))["bayrak"])}
        satir.append(s)
        print(json.dumps({a: s[a] for a in ("ses", "ton", *[f"b{b:g}" for b in betalar])}, ensure_ascii=False), flush=True)
    kipler = [f"b{b:g}" for b in betalar] + ["ogretmen"]
    ozet = {kip: toplu(satir, kip) for kip in kipler}
    print(json.dumps(ozet, ensure_ascii=False, indent=1))
    (CIKTI / "cesitlilik_sonuc.json").write_text(json.dumps(satir, ensure_ascii=False, indent=1), encoding="utf-8")
    (CIKTI / "cesitlilik_ozet.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=1), encoding="utf-8")
    ad = {kip: (f"β {kip[1:]}" + (" (şimdiki)" if kip == "b0.4" else "")) for kip in kipler[:-1]} | {"ogretmen": "Öğretmen"}
    tablo = "".join(f"<tr><td>{ad[k]}</td>" + "".join(f"<td>{v.get(a, '—')}</td>" for a in
                    ("duzey_sapmasi", "duzey_r", "ic_sapma", "benzerlik", "mcd", "utmos", "takilan")) + "</tr>"
                    for k, v in ozet.items())
    satirlar = "".join(
        f"<tr><td><b>{s['ses']}</b> · {s['ton']}<br>{html.escape(ipa_metin(s['okunus']))}</td>"
        + "".join(f"<td><audio controls preload=none src='{s['ses']}/{Path(s['dosya']).stem}_{k}.wav'></audio>"
                  + (" <b>takıldı</b>" if s[k].get("takildi") else "") + "</td>" for k in kipler) + "</tr>"
        for s in satir)
    (CIKTI / "cesitlilik.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Perde çeşitliliği</title><style>body{font:14px sans-serif;"
        "margin:16px}td,th{padding:4px 8px;border-bottom:1px solid #ddd;vertical-align:top;text-align:left}audio{height"
        ":30px;width:190px}</style><h1>Perde çeşitliliği: difüzyon stilinin payı (β)</h1><p>düzey sapması: aynı sesin "
        "cümleleri arasında perde düzeyinin değişimi (yarım ton; öğretmen 3,4). düzey r: cümleye özgü düzeyin "
        "öğretmeninkiyle ilişkisi.</p><table><tr><th>okuyuş</th><th>düzey sapması</th><th>düzey r</th><th>cümle içi"
        "</th><th>benzerlik</th><th>MCD</th><th>UTMOS</th><th>takılan</th></tr>" + tablo + "</table><h2>Dinle</h2>"
        "<table><tr><th>cümle</th>" + "".join(f"<th>{ad[k]}</th>" for k in kipler) + "</tr>" + satirlar + "</table>",
        encoding="utf-8")
    print(CIKTI / "cesitlilik.html")


if __name__ == "__main__":
    b = sys.argv[sys.argv.index("--beta") + 1] if "--beta" in sys.argv else "0.4,0.7,1.0"
    main([float(x) for x in b.split(",")])
