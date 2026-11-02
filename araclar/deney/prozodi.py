# -*- coding: utf-8 -*-
"""
EZGİ BANKASI ÖLÇÜMÜ (02.11) — ogrenci/ezgi.py'nin README'deki ölçümü. t10 doğrulama listesindeki (eğitimde ve bankada
olmayan) cümleler, 6 ses × 5 cümle. Dört okuyuş:
  ogrenci     öğrencinin kendi ezgisi
  ust_sinir   öğrenci + aynı cümlenin öğretmen ezgisi (bankada aynı cümle varsa stüdyonun "tam" kipi bunu verir)
  banka       öğrenci + bankadaki en benzer başka cümlenin ezgisi ("benzer" kipi)
  ogretmen    öğretmenin kaydı
Ölçü: hece perdesi ve süresinin öğretmenle ilişkisi (r; öğretmenin kaydı öğrencinin hizalayıcısıyla hecelere
bölünür), CTC denetimi (takılan), UTMOS. Çıktı: cikti/prozodi/ (sonuc.json, ozet.json, dinle.html).

    python araclar/deney/prozodi.py [--adet 5]        (önce: python ogrenci/ezgi.py banka)
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "araclar" / "olcu")]
CIKTI = KOK / "cikti" / "prozodi"
LISTE = KOK / "veri" / "egitim_vurgu_t10"
SESLER = ["damla", "kuzey", "zeynep", "serdar", "fatma", "emre"]
ORAN = 24000
_IPA = [("tʃ", "ç"), ("dʒ", "c"), ("ʃ", "ş"), ("ʒ", "j"), ("ɯ", "ı"), ("ɾ", "r"), ("ɫ", "l"), ("ɟ", "g"), ("ɡ", "g"),
        ("c", "k"), ("ø", "ö"), ("y", "ü"), ("j", "y"), ("ɣ", "ğ"), ("ɛ", "e"), ("ɑ", "a"), ("ˈ", ""), ("ː", "")]


def ipa_metin(okunus: str) -> str:
    """CTC denetimi için okunuştan yaklaşık Türkçe yazım (bütün okuyuşlar aynı başvuruyla ölçülür)."""
    for a, b in _IPA:
        okunus = okunus.replace(a, b)
    return " ".join("".join(c for c in w if c.isalpha()) for w in okunus.split())


def isaret_sureleri(ogr, okunus: str, stil: np.ndarray, ezgi: dict | None) -> np.ndarray:
    """model.konus'un kullandığı işaret süreleri (ezgi varsa öğretmenin hece süreleriyle uzatılmış)."""
    from ezgi import sure_uzat
    hiza = ogr.hazirla("", stil, okunus=okunus)[2]
    if ezgi is not None:
        hiza = ogr.hazirla("", stil, okunus=okunus, uzat=sure_uzat(ogr, okunus, hiza[0].sum(-1).long().numpy(), ezgi))[2]
    return hiza[0].sum(-1).long().numpy()


def _r(a: np.ndarray, b: np.ndarray) -> float:
    k = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() >= 3 and a[k].std() > 0 and b[k].std() > 0 else np.nan


def main(adet: int = 5) -> None:
    import soundfile as sf
    import denetim
    from dogallik import puan
    from ezgi import WAVS, Banka, cikar, hece_ezgisi, heceler, perde
    from model import Ogrenci
    from studyo import URUN_AGIRLIK
    ogr = Ogrenci(KOK / "cikti" / "ogrenci" / URUN_AGIRLIK)
    sozluk = ogr.temiz.word_index_dictionary
    tonlar = json.loads((KOK / "veri" / "egitim" / "ozet.json").read_text(encoding="utf-8"))["duygu"]
    banka = Banka()
    val = [s.split("|") for s in (LISTE / "val_list.txt").read_text(encoding="utf-8").splitlines() if s]
    sonuc = []
    for ses in SESLER:
        (CIKTI / ses).mkdir(parents=True, exist_ok=True)
        ogr.stil_dosyadan(ses)
        for dosya, okunus, _ in [s for s in val if s[0].startswith(ses + "/") and len(heceler(s[1], sozluk)[0]) >= 4][:adet]:
            ad, ton = Path(dosya).stem, tonlar.get(dosya.rsplit(".", 1)[0], "notr")
            stil = ogr.cumle_stili(okunus, ogr.stil_dosyadan(ses, ton if f"{ses}/{ton}" in ogr.stiller else "notr"),
                                   okunus=okunus)
            x, _ = sf.read(WAVS / dosya, dtype="float32")
            c = cikar(ogr, x, okunus)
            if c is None:
                continue
            ogretmen = hece_ezgisi(okunus, c["sure"], c["f0"], sozluk)
            hedef = np.nanmean(ogretmen["cizgi"], axis=1)
            benzer, bilgi = banka.bul(ses, okunus, ton, "benzer", sozluk, ayniyi_atla=True)
            sf.write(CIKTI / ses / f"{ad}_ogretmen.wav", x, ORAN)
            kayit = {"ses": ses, "dosya": dosya, "okunus": okunus, "ton": ton, "bulunan": bilgi.get("kaynak"),
                     "uzaklik": bilgi.get("uzaklik")}
            ref = ipa_metin(okunus)
            for kip, ezgi in (("ogrenci", None), ("ust_sinir", {"sure": ogretmen["sure"] / ogretmen["sure"].mean(),
                                                               "cizgi": ogretmen["cizgi"]}), ("banka", benzer)):
                w = ogr.konus("", stil, okunus=okunus, ezgi=ezgi)
                yol = CIKTI / ses / f"{ad}_{kip}.wav"
                sf.write(yol, w, ORAN)
                e = hece_ezgisi(okunus, isaret_sureleri(ogr, okunus, stil, ezgi), perde(ogr, w), sozluk)
                with np.errstate(all="ignore"):
                    kayit[kip] = {"perde_r": round(_r(np.nanmean(e["cizgi"], axis=1), hedef), 3),
                                  "sure_r": round(_r(e["sure"], ogretmen["sure"]), 3),
                                  "takildi": bool(denetim.denetle(w, ORAN, ref)["bayrak"]), "utmos": round(puan(yol), 2)}
            kayit["ogretmen"] = {"takildi": bool(denetim.denetle(x, ORAN, ref)["bayrak"]),
                                 "utmos": round(puan(CIKTI / ses / f"{ad}_ogretmen.wav"), 2)}
            sonuc.append(kayit)
            print(json.dumps({k: kayit[k] for k in ("ses", "ton", "uzaklik", "ogrenci", "ust_sinir", "banka")},
                             ensure_ascii=False), flush=True)
    (CIKTI / "sonuc.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    ozet = {}
    for kip in ("ogrenci", "ust_sinir", "banka", "ogretmen"):
        v = [k[kip] for k in sonuc]
        ortalama = lambda a: round(float(np.nanmean([x.get(a, np.nan) for x in v])), 3) if kip != "ogretmen" else 1.0
        ozet[kip] = {"perde_r": ortalama("perde_r"), "sure_r": ortalama("sure_r"), "takilan": sum(x["takildi"] for x in v),
                     "utmos": round(float(np.mean([x["utmos"] for x in v])), 2), "cumle": len(v)}
    print(json.dumps(ozet, ensure_ascii=False, indent=1))
    (CIKTI / "ozet.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=1), encoding="utf-8")
    sayfa(sonuc, ozet)


def sayfa(sonuc: list[dict], ozet: dict) -> None:
    ad = {"ogrenci": "Öğrenci", "ust_sinir": "Öğrenci + aynı cümlenin öğretmen ezgisi",
          "banka": "Öğrenci + benzer cümlenin ezgisi", "ogretmen": "Öğretmen"}
    satir = []
    for k in sonuc:
        a = Path(k["dosya"]).stem
        hucre = "".join(f"<td><audio controls preload=none src='{k['ses']}/{a}_{kip}.wav'></audio><br><small>"
                        + (f"perde r {k[kip]['perde_r']} · süre r {k[kip]['sure_r']} · " if kip != "ogretmen" else "")
                        + f"UTMOS {k[kip]['utmos']}{' · <b>takıldı</b>' if k[kip]['takildi'] else ''}</small></td>"
                        for kip in ad)
        satir.append(f"<tr><td><b>{k['ses']}</b> · {k['ton']}<br>{html.escape(ipa_metin(k['okunus']))}<br>"
                     f"<small>benzer: {html.escape(str(k['bulunan']))}</small></td>{hucre}</tr>")
    tablo = "".join(f"<tr><td>{ad[k]}</td><td>{v['perde_r']}</td><td>{v['sure_r']}</td><td>{v['utmos']}</td>"
                    f"<td>{v['takilan']} / {v['cumle']}</td></tr>" for k, v in ozet.items())
    (CIKTI / "dinle.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Ezgi bankası</title><style>body{font:14px sans-serif;margin:16px}"
        "td,th{padding:4px 8px;border-bottom:1px solid #ddd;vertical-align:top;text-align:left}audio{height:30px;"
        "width:210px}</style><h1>Ezgi bankası: öğrencinin sesi, öğretmenin ezgisi</h1><p>Eğitimde görülmemiş "
        "cümleler. perde r / süre r: hece perdesi ve süresinin öğretmenle ilişkisi (1 = aynı ezgi).</p><table><tr>"
        "<th>okuyuş</th><th>perde r</th><th>süre r</th><th>UTMOS</th><th>takılan</th></tr>" + tablo + "</table><h2>Dinle"
        "</h2><table><tr><th>cümle</th>" + "".join(f"<th>{v}</th>" for v in ad.values()) + "</tr>" + "".join(satir) +
        "</table>", encoding="utf-8")
    print(CIKTI / "dinle.html")


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--adet") + 1]) if "--adet" in sys.argv else 5)
