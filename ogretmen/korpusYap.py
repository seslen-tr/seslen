# -*- coding: utf-8 -*-
"""
ÖĞRETMEN DERLEMİ — okunacak cümleler (veri/korpus.jsonl).

Kaynaklar (hepsi ticari kullanıma açık):
  cv      Common Voice Türkçe cümle havuzu (sentence-collector + wiki), CC0.
          github.com/common-voice/common-voice/tree/main/server/data/tr
  urun    Portal / Seslen'in uygulama cümleleri (bizim metnimiz): karakter
          ürünün asıl cümlelerini de duymuş olsun.
  seslen  Seslen'in eski derlemi (900 kısa cümle, bizim metnimiz).

Seçim: süzgeç (4-18 sözcük, rakamsız, kısaltmasız, yalnız Türkçe harf) →
tekrarsız → SES KAPSAMI açgözlü: Türkçe yazı sese yakın olduğu için harf
ikilileri (ve sözcük sınırı) sayılır; her adımda yeni ikili en çok getiren
cümle alınır, sonra rastgele doldurulur. Soru ve ünlem payı korunur
(tonlama çeşitliliği).

    python ogretmen/korpusYap.py --adet 6000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
VERI = KOK / "veri"
SESLEN = KOK.parent / "seslen" / "ses"
CV = "https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/tr/"
CV_DOSYALAR = ("sentence-collector.txt", "wiki.tr.txt")

HARF = "abcçdefgğhıijklmnoöprsştuüvyzâîû"
IZINLI = re.compile(rf"^[{HARF}{HARF.upper()}İ .,;:!?'’\-\"]+$")


def indir() -> list[tuple[str, str]]:
    d = VERI / "cv-metin"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for ad in CV_DOSYALAR:
        yol = d / ad
        if not yol.exists():
            print(f"indiriliyor {ad}…", flush=True)
            urllib.request.urlretrieve(CV + ad, yol)
        out += [("cv", s.strip()) for s in yol.read_text(encoding="utf-8").splitlines() if s.strip()]
    return out


def bizim() -> list[tuple[str, str]]:
    out = []
    u = SESLEN / "onUretim-uygulamaCumleleri.json"
    if u.exists():
        out += [("urun", c["metin"]) for c in json.loads(u.read_text(encoding="utf-8"))["cumleler"]]
    k = SESLEN / "damitma" / "korpus.json"
    if k.exists():
        out += [("seslen", c["metin"]) for c in json.loads(k.read_text(encoding="utf-8"))]
    return out


def uygun(s: str) -> bool:
    s = s.replace("’", "'")
    if not IZINLI.match(s) or not (20 <= len(s) <= 140):
        return False
    w = s.split()
    if not (4 <= len(w) <= 18):
        return False
    if any(sum(ch.isupper() for ch in x) >= 2 for x in w):     # kısaltma / BÜYÜK HARF
        return False
    return s[-1] in ".!?"


def ikililer(s: str) -> set[str]:
    t = " " + unicodedata.normalize("NFC", s.replace("I", "ı").replace("İ", "i").lower()) + " "
    t = re.sub(rf"[^{HARF} ]", "", t)
    t = re.sub(" +", " ", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def sec(havuz: list[tuple[str, str]], adet: int, tohum: int = 1) -> list[tuple[str, str]]:
    rnd = random.Random(tohum)
    rnd.shuffle(havuz)
    secilen, gorulen = [], Counter()
    # 1) kapsam: en çok yeni ikili getiren (tembel açgözlü, örneklemle hızlı)
    kalan = havuz[:]
    while len(secilen) < adet * 0.3 and kalan:
        ornek = kalan[:400]
        en = max(ornek, key=lambda c: sum(1 / (1 + gorulen[b]) for b in ikililer(c[1])))
        secilen.append(en)
        gorulen.update(ikililer(en[1]))
        kalan.remove(en)
    # 2) doldur: tür payını koruyarak rastgele
    soru = [c for c in kalan if c[1].endswith("?")]
    unlem = [c for c in kalan if c[1].endswith("!")]
    duz = [c for c in kalan if c[1].endswith(".")]
    for grup, pay in ((soru, 0.15), (unlem, 0.08)):
        secilen += grup[: int(adet * pay)]
    secilen += duz[: adet - len(secilen)]
    return secilen[:adet], len(gorulen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adet", type=int, default=6000, help="toplam cümle (bizimkiler dahil)")
    a = ap.parse_args()
    bizimkiler = bizim()
    cv = [c for c in indir() if uygun(c[1])]
    ayni = {c[1] for c in bizimkiler}
    cv = list({c[1]: c for c in cv if c[1] not in ayni}.values())
    print(f"havuz: cv {len(cv)} uygun cümle, bizim {len(bizimkiler)}")
    secilen, kapsam = sec(cv, a.adet - len(bizimkiler))
    tum = bizimkiler + secilen
    VERI.mkdir(exist_ok=True)
    with open(VERI / "korpus.jsonl", "w", encoding="utf-8") as f:
        for i, (kaynak, metin) in enumerate(tum, 1):
            f.write(json.dumps({"kimlik": f"k{i:05d}", "metin": metin, "kaynak": kaynak}, ensure_ascii=False) + "\n")
    tur = Counter(m[-1] for _, m in tum)
    sozcuk = sum(len(m.split()) for _, m in tum)
    print(f"korpus: {len(tum)} cümle, {sozcuk} sözcük (~{sozcuk / 2.6 / 3600:.1f} saat ses), "
          f"ikili kapsamı {kapsam}, tür {dict(tur)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
