# -*- coding: utf-8 -*-
"""
PERDE DÜZEYİ VE GENİŞLİĞİ (02-03.11) — doku.py'nin bulgusu: öğrenciyi öğretmenden ayıran perdenin düzeyi.
  - Öğrenci cümleyi öğretmenden ortalama 3,3 yarım ton farklı düzeyde okuyor (Kuzey +3,9, Serdar +3,5).
  - Cümleye özgü düzeyi öğretmenle r ≈ 0,5 ilişkili (yönü doğru), ama değişimi yarı genişlikte: cümleler arası
    sapma öğrencide 1,9, öğretmende 3,4 yarım ton (ortalamaya eğitilen tahmincinin düzleşmesi; cesitlilik.py).
Eğitimsiz iki düzeltme, ses × ton istatistikleri t10 eğitim cümlelerinden (sınama cümleleri değil):
  duzey   sabit kayma: öğrencinin ortalama düzeyi öğretmeninkine (02.11; kulak: "bazısı iyi bazısı kötü")
  gv      küresel varyans (GV) düzeltmesi: hedef = μ_öğretmen + k·(cümlenin düzeyi − μ_öğrenci), k = σ_öğretmen /
          σ_öğrenci (ses başına); cümle içi perde genişliği de öğretmenin oranına (tonlama çarpanı)
Sınama: doku.py'nin 30 cümlesi; öğrenci ve öğrenci + öğretmen ezgisi ("tam") düzeltmeli ve düzeltmesiz.

    python araclar/deney/duzey.py [--olc 30]      → cikti/doku/duzey_egitim.json, duzey_ozet.json, duzey.html
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
EN_AZ = 5                     # ses × ton ortalaması için en az cümle; azsa sesin ortalaması
K_ARALIK = (1.0, 2.5)
AD = {"ogrenci": "Öğrenci", "ust_sinir": "Öğrenci + öğretmen ezgisi", "ogrenci_duzey": "Öğrenci, sabit kayma",
      "tam_duzey": "Öğrenci + ezgi, sabit kayma", "ogrenci_gv": "Öğrenci, GV", "tam_gv": "Öğrenci + ezgi, GV",
      "ogretmen": "Öğretmen"}


def egitim_olcumu(ogr, tonlar: dict, sesler: list[str], olc: int) -> list[dict]:
    """Eğitim cümlelerinde öğrencinin ve öğretmenin perde düzeyi ve cümle içi sapması (yarım ton)."""
    import soundfile as sf
    from cesitlilik import perde_ozeti
    from ezgi import WAVS
    from prozodi import LISTE
    satir = [s.split("|") for s in (LISTE / "train_list.txt").read_text(encoding="utf-8").splitlines() if s]
    rng, sonuc = np.random.default_rng(0), []
    for ses in sesler:
        aday = [s for s in satir if s[0].startswith(ses + "/") and (WAVS / s[0]).exists()]
        sayac: dict[str, int] = {}
        for dosya, okunus, _ in [aday[i] for i in rng.permutation(len(aday))[: olc * 3]]:
            ton = tonlar.get(dosya.rsplit(".", 1)[0], "notr")
            if sayac.get(ton, 0) >= olc:
                continue
            stil = ogr.cumle_stili(okunus, ogr.stil_dosyadan(ses, ton if f"{ses}/{ton}" in ogr.stiller else "notr"),
                                   okunus=okunus)
            x, _ = sf.read(WAVS / dosya, dtype="float32")
            (sd, si), (td, ti) = perde_ozeti(ogr, ogr.konus("", stil, okunus=okunus)), perde_ozeti(ogr, x)
            if np.isfinite([sd, si, td, ti]).all():
                sonuc.append({"ses": ses, "ton": ton, "s_duz": sd, "t_duz": td, "s_ic": si, "t_ic": ti})
                sayac[ton] = sayac.get(ton, 0) + 1
    return sonuc


def parametreler(satir: list[dict]) -> dict:
    """Ses × ton ortalamaları (μ_öğrenci, μ_öğretmen), ses başına k ve cümle içi genişlik oranı."""
    p: dict = {}
    for ses in sorted({s["ses"] for s in satir}):
        v = [s for s in satir if s["ses"] == ses]
        ort = {"": (np.mean([s["s_duz"] for s in v]), np.mean([s["t_duz"] for s in v]))}
        for ton in {s["ton"] for s in v}:
            w = [s for s in v if s["ton"] == ton]
            if len(w) >= EN_AZ:
                ort[ton] = (np.mean([s["s_duz"] for s in w]), np.mean([s["t_duz"] for s in w]))
        mu = lambda s: ort.get(s["ton"], ort[""])
        sapma_s = np.std([s["s_duz"] - mu(s)[0] for s in v])
        sapma_t = np.std([s["t_duz"] - mu(s)[1] for s in v])
        p[ses] = {"ort": {t: [round(float(a), 2), round(float(b), 2)] for t, (a, b) in ort.items()},
                  "k": round(float(np.clip(sapma_t / max(sapma_s, 1e-6), *K_ARALIK)), 2),
                  "ic": round(float(np.mean([s["t_ic"] for s in v]) / np.mean([s["s_ic"] for s in v])), 2),
                  "adet": len(v)}
    return p


def duzeltme(p: dict, ses: str, ton: str, duzey: float, gv: bool) -> tuple[float, float]:
    """(perde çarpanı, tonlama çarpanı): cümlenin öğrenci düzeyi → öğretmenin ölçeği."""
    ms, mt = p[ses]["ort"].get(ton, p[ses]["ort"][""])
    hedef = mt + (p[ses]["k"] * (duzey - ms) if gv else duzey - ms)
    return 2 ** ((hedef - duzey) / 12), (p[ses]["ic"] if gv else 1.0)


def main(olc: int = 30) -> None:
    import soundfile as sf
    from cesitlilik import perde_ozeti, toplu
    from doku import Benzerlik, kepstrum, mcd
    from dogallik import puan
    from ezgi import cikar, hece_ezgisi
    from model import Ogrenci
    from studyo import URUN_AGIRLIK
    ogr = Ogrenci(KOK / "cikti" / "ogrenci" / URUN_AGIRLIK)
    sozluk = ogr.temiz.word_index_dictionary
    tonlar = json.loads((KOK / "veri" / "egitim" / "ozet.json").read_text(encoding="utf-8"))["duygu"]
    sinama = json.loads((CIKTI / "sonuc.json").read_text(encoding="utf-8"))
    sesler = list(dict.fromkeys(k["ses"] for k in sinama))
    yol = CIKTI / "duzey_egitim.json"
    egitim = json.loads(yol.read_text(encoding="utf-8")) if yol.exists() else egitim_olcumu(ogr, tonlar, sesler, olc)
    yol.write_text(json.dumps(egitim, ensure_ascii=False, indent=0), encoding="utf-8")
    p = parametreler(egitim)
    (CIKTI / "duzey_parametre.json").write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({s: {"k": v["k"], "ic": v["ic"], "ort": v["ort"][""]} for s, v in p.items()}), flush=True)
    sv, sonuc = Benzerlik(), []
    for k in sinama:
        ses, ton, okunus, ad = k["ses"], k["ton"], k["okunus"], Path(k["dosya"]).stem
        stil = ogr.cumle_stili(okunus, ogr.stil_dosyadan(ses, ton if f"{ses}/{ton}" in ogr.stiller else "notr"),
                               okunus=okunus)
        x, _ = sf.read(CIKTI / ses / f"{ad}_ogretmen.wav", dtype="float32")
        c = cikar(ogr, x, okunus)
        e = hece_ezgisi(okunus, c["sure"], c["f0"], sozluk)
        ezgi = {"sure": e["sure"] / e["sure"].mean(), "cizgi": e["cizgi"]}
        g, kp = sv.gomme(x), kepstrum(ogr, x)
        d0, i0 = perde_ozeti(ogr, x)
        satir = {"ses": ses, "ton": ton, "dosya": k["dosya"], "okunus": okunus, "ogretmen": {"duzey": d0, "ic": i0}}
        dalga = {kip: sf.read(CIKTI / ses / f"{ad}_{kip}.wav", dtype="float32")[0] for kip in ("ogrenci", "ust_sinir")}
        for temel, ez in (("ogrenci", None), ("ust_sinir", ezgi)):
            duz = perde_ozeti(ogr, dalga[temel])[0]
            for ek, gv in (("duzey", False), ("gv", True)):
                carpan, tonlama = duzeltme(p, ses, ton, duz, gv)
                kip = ("ogrenci_" if temel == "ogrenci" else "tam_") + ek
                dalga[kip] = ogr.konus("", stil, okunus=okunus, perde=carpan, tonlama=tonlama, ezgi=ez)
                satir[kip + "_carpan"] = round(carpan, 3)
        for kip, w in dalga.items():
            yol_w = CIKTI / ses / f"{ad}_{kip}.wav"
            sf.write(yol_w, w, ORAN)
            d, i = perde_ozeti(ogr, w)
            satir[kip] = {"duzey": d, "ic": i, "benzerlik": round(float(sv.gomme(w) @ g), 3), "utmos": round(puan(yol_w), 2),
                          "mcd": round(mcd(kepstrum(ogr, w), kp), 2), "duzey_farki": round(abs(d - d0), 2), "takildi": False}
        sonuc.append(satir)
        print(json.dumps({a: satir[a]["benzerlik"] for a in dalga} | {"ses": ses}, ensure_ascii=False), flush=True)
    kipler = ["ogrenci", "ust_sinir", "ogrenci_duzey", "tam_duzey", "ogrenci_gv", "tam_gv", "ogretmen"]
    ozet = {kip: toplu(sonuc, kip) for kip in kipler}
    for kip in kipler[:-1]:
        ozet[kip].pop("takilan", None)
        ozet[kip]["duzey_farki"] = round(float(np.mean([s[kip]["duzey_farki"] for s in sonuc])), 2)
    print(json.dumps(ozet, ensure_ascii=False, indent=1))
    (CIKTI / "duzey_sonuc.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    (CIKTI / "duzey_ozet.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=1), encoding="utf-8")
    sayfa(sonuc, ozet, kipler)


def sayfa(sonuc: list[dict], ozet: dict, kipler: list[str]) -> None:
    from prozodi import ipa_metin
    alan = ("duzey_farki", "duzey_sapmasi", "duzey_r", "ic_sapma", "benzerlik", "mcd", "utmos")
    tablo = "".join(f"<tr><td>{AD[k]}</td>" + "".join(f"<td>{v.get(a, '—')}</td>" for a in alan) + "</tr>"
                    for k, v in ozet.items())
    satir = []
    for s in sonuc:
        a = Path(s["dosya"]).stem
        hucre = "".join(f"<td><audio controls preload=none src='{s['ses']}/{a}_{kip}.wav'></audio><br><small>"
                        + (f"benzerlik {s[kip]['benzerlik']} · düzey farkı {s[kip]['duzey_farki']}"
                           if "benzerlik" in s[kip] else "") + "</small></td>" for kip in kipler)
        satir.append(f"<tr><td><b>{s['ses']}</b> · {s['ton']}<br>{html.escape(ipa_metin(s['okunus']))}</td>{hucre}</tr>")
    (CIKTI / "duzey.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Perde düzeyi</title><style>body{font:14px sans-serif;margin:16px}"
        "td,th{padding:4px 8px;border-bottom:1px solid #ddd;vertical-align:top;text-align:left}audio{height:30px;"
        "width:170px}</style><h1>Perde düzeyi: sabit kayma ve GV düzeltmesi</h1><p>düzey farkı: öğretmenin ortanca "
        "perdesinden uzaklık (yarım ton). düzey sapması: aynı sesin cümleleri arasında düzeyin değişimi (öğretmen "
        "3,4). düzey r: cümleye özgü düzeyin öğretmeninkiyle ilişkisi. benzerlik: konuşmacı benzerliği.</p><table>"
        "<tr><th>okuyuş</th><th>düzey farkı</th><th>düzey sapması</th><th>düzey r</th><th>cümle içi</th>"
        "<th>benzerlik</th><th>MCD</th><th>UTMOS</th></tr>" + tablo + "</table><h2>Dinle</h2><table><tr><th>cümle</th>"
        + "".join(f"<th>{AD[k]}</th>" for k in kipler) + "</tr>" + "".join(satir) + "</table>", encoding="utf-8")
    print(CIKTI / "duzey.html")


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--olc") + 1]) if "--olc" in sys.argv else 30)
