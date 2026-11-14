# -*- coding: utf-8 -*-
"""
EZGİCİ ÖLÇÜMÜ — doku.py'nin 30 görülmemiş cümlesinde (cikti/doku/sonuc.json), cesitlilik.py'nin ölçüleriyle:
düzey sapması (öğretmen 3,4), düzeyin öğretmenle ilişkisi, cümle içi perde sapması, öğretmene benzerlik (WavLM-SV), MCD,
UTMOS, takılan (CTC). Örnekleme r'yi düşürür, bu beklenen bir şey: örnek, öğretmenin o cümledeki tek okuyuşunu değil
dağılımını hedefler; karar çeşitlilik + benzerlik + kulakla verilir. Çıktı: cikti/ezgici/olcum/ (ozet.json, dinle.html).

    python ogrenci/ezgici/olc.py [--kip ogrenci,e1,e07,e05]
    python ogrenci/ezgici/olc.py --sayfa        (yalnız dinleme sayfası, ölçülmüş sonuçtan)
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "ogrenci" / "ezgici"), str(KOK / "araclar" / "olcu"),
                str(KOK / "araclar" / "deney"), str(KOK / "ogretmen")]
CIKTI = KOK / "cikti" / "ezgici" / "olcum"
DOKU = KOK / "cikti" / "doku"
ORAN = 24000
# kip → sıcaklık; None: öğrencinin kendisi. Süre örneklemesi (04.11'de e1s / e07s) ölçüde bir şey kazandırmadı.
KIPLER = {"ogrenci": None, "e1": 1.0, "e07": 0.7, "e05": 0.5, "sabit": None, "sabit_e07": 0.7}
SABIT = {"sabit", "sabit_e07"}           # cümleye özgü difüzyon stili yok (stüdyoda difuzyon=False, seslen-tts gibi)
AD = {"ogrenci": "Öğrenci", "e1": "Ezgici, tam örnek", "e07": "Ezgici, sıcaklık 0,7 (ürün)", "e05": "Ezgici, sıcaklık 0,5",
      "sabit": "Sabit stil", "sabit_e07": "Sabit stil + ezgici 0,7", "ogretmen": "Öğretmen"}
DINLE = ("ogrenci", "e07", "sabit", "ogretmen")
SUTUN = {"duzey_sapmasi": "düzey sapması", "duzey_r": "düzey r", "ic_sapma": "cümle içi", "benzerlik": "benzerlik",
         "mcd": "MCD", "utmos": "UTMOS", "takilan": "takılan"}


def main(kipler: list[str]) -> None:
    import denetim
    import soundfile as sf
    from cesitlilik import perde_ozeti, toplu
    from doku import Benzerlik, kepstrum, mcd
    from dogallik import puan
    from model import Ogrenci
    from prozodi import ipa_metin
    from studyo import URUN_AGIRLIK
    from uygula import Ezgi
    from yukle import sureci_hazirla
    sureci_hazirla()
    ogr = Ogrenci(KOK / "cikti" / "ogrenci" / URUN_AGIRLIK)
    ezgi = Ezgi()
    sv, satir = Benzerlik(), []
    for k in json.loads((DOKU / "sonuc.json").read_text(encoding="utf-8")):
        ses, ton, okunus, ad = k["ses"], k["ton"], k["okunus"], Path(k["dosya"]).stem
        ogr.stil_dosyadan(ses)
        sabit = ogr.stil_dosyadan(ses, ton if f"{ses}/{ton}" in ogr.stiller else "notr")
        stil = ogr.cumle_stili(okunus, sabit, okunus=okunus)
        (CIKTI / ses).mkdir(parents=True, exist_ok=True)
        x, _ = sf.read(DOKU / ses / f"{ad}_ogretmen.wav", dtype="float32")
        sf.write(CIKTI / ses / f"{ad}_ogretmen.wav", x, ORAN)
        g, kp = sv.gomme(x), kepstrum(ogr, x)
        d0, i0 = perde_ozeti(ogr, x)
        s = {"ses": ses, "ton": ton, "dosya": k["dosya"], "okunus": okunus, "ogretmen": {"duzey": d0, "ic": i0}}
        for kip in kipler:
            ezgici = None if KIPLER[kip] is None else ezgi.cagir(ses, ton, ogr.temiz.word_index_dictionary, KIPLER[kip])
            w = ogr.konus("", sabit if kip in SABIT else stil, okunus=okunus, ezgici=ezgici)
            p = CIKTI / ses / f"{ad}_{kip}.wav"
            sf.write(p, w, ORAN)
            d, i = perde_ozeti(ogr, w)
            s[kip] = {"duzey": d, "ic": i, "benzerlik": round(float(sv.gomme(w) @ g), 3), "utmos": round(puan(p), 2),
                      "mcd": round(mcd(kepstrum(ogr, w), kp), 2),
                      "takildi": bool(denetim.denetle(w, ORAN, ipa_metin(okunus))["bayrak"])}
        satir.append(s)
        print(json.dumps({a: s[a] for a in ("ses", "ton", *kipler)}, ensure_ascii=False), flush=True)
    hepsi = [*kipler, "ogretmen"]
    ozet = {kip: toplu(satir, kip) for kip in hepsi}
    print(json.dumps(ozet, ensure_ascii=False, indent=1))
    (CIKTI / "sonuc.json").write_text(json.dumps(satir, ensure_ascii=False, indent=1), encoding="utf-8")
    (CIKTI / "ozet.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=1), encoding="utf-8")
    sayfa(satir, ozet)


def sayfa(satir: list[dict], ozet: dict) -> None:
    from prozodi import ipa_metin
    dinle = [k for k in DINLE if k in ozet]
    tablo = "".join(f"<tr><td>{AD[k]}</td>" + "".join(f"<td>{v.get(a, '—')}</td>" for a in SUTUN) + "</tr>"
                    for k, v in ozet.items())
    satirlar = "".join(
        f"<tr><td><b>{s['ses']}</b> · {s['ton']}<br>{html.escape(ipa_metin(s['okunus']))}</td>"
        + "".join(f"<td><audio controls preload=none src='{s['ses']}/{Path(s['dosya']).stem}_{k}.wav'></audio>"
                  + (" <b>takıldı</b>" if s[k].get("takildi") else "") + "</td>" for k in dinle) + "</tr>"
        for s in satir)
    (CIKTI / "dinle.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Ezgici</title><style>body{font:14px sans-serif;margin:16px}"
        "td,th{padding:4px 8px;border-bottom:1px solid #ddd;vertical-align:top;text-align:left}audio{height:30px;"
        "width:190px}</style><h1>Ezgici: öğretmenden örneklenen ezgi</h1><p>Görülmemiş cümleler. Ezgici öğretmenin bu "
        "cümledeki okuyuşunu değil, onun gibi bir okuyuşu üretir; aynı ezgiyi aramayın, canlılığına bakın.</p><table><tr>"
        "<th>okuyuş</th>" + "".join(f"<th>{v}</th>" for v in SUTUN.values()) + "</tr>" + tablo + "</table><h2>Dinle</h2>"
        "<table><tr><th>cümle</th>" + "".join(f"<th>{AD[k]}</th>" for k in dinle) + "</tr>" + satirlar + "</table>",
        encoding="utf-8")
    print(CIKTI / "dinle.html")


if __name__ == "__main__":
    if "--sayfa" in sys.argv:
        sayfa(json.loads((CIKTI / "sonuc.json").read_text(encoding="utf-8")),
              json.loads((CIKTI / "ozet.json").read_text(encoding="utf-8")))
        sys.exit()
    k = sys.argv[sys.argv.index("--kip") + 1] if "--kip" in sys.argv else "ogrenci,e1,e07,e05"
    main(k.split(","))
