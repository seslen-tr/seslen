# -*- coding: utf-8 -*-
"""
ÖN UÇ SINAVI (01.11) — Seslen'in metin yolunu kardeş projenin açık sınavıyla (turkce-tts-sinav) ölçer. Modelsiz,
PyTorch'suz; GitHub'da her gönderimde çalışır (.github/workflows/test.yml): kardeş kütüphanelerden biri yeni sürüme
geçince Seslen'in metin yolu bozulursa orada görünür.

Ölçülen, stüdyonun gerçekten kullandığı işlevler (kullanıcının telaffuz sözlüğü hariç):
  metin hazırlama  turkce_vurgu.normallestir (turkce-normalize + yabancı adlar; küçük harfe çevirir, model küçük
                   harfle çalışır: büyük-küçük harf ve noktalama farkı sayılmaz)
  vurgu            turkce_vurgu.vurgu.vurgulu_hece
  okunuş           turkce_vurgu.g2p.sozcuk_okunusu

    python araclar/olcu/onuc.py [--esik]      --esik: eşiğin altındaysa çıkış kodu 1
"""
from __future__ import annotations

import re
import sys

# 01.11 ölçümü: metin hazırlama 126/128 ("No:" → "numara", "Ar-Ge" → "ar, ge"), literatür kural+Sezer 120/123,
# okunuş PER %3,04. Eşikler küçük bir pay bırakır; düşüş yeni bir hata demektir.
ESIK = {"hazirlama_fark": 3, "vurgu_literatur": 0.95, "okunus_per": 0.035}


def olc(yaz: bool = True) -> dict:
    from turkce_tts_sinav import kucuk, normalizasyon_sinavi, okunus_sinavi, vurgu_sinavi
    from turkce_vurgu import normallestir
    from turkce_vurgu.g2p import sozcuk_okunusu
    from turkce_vurgu.vurgu import vurgulu_hece

    # tire sözcük sınırıdır (seste "Ar-Ge" = "ar ge"); öteki noktalama sayılmaz
    sozcukler = lambda t: re.sub(r"[^\w\s]", "", re.sub(r"[-–‐‑]", " ", kucuk(t))).split()
    n = normalizasyon_sinavi(normallestir, yaz=False)
    fark = [y for y in n["yanlislar"] if sozcukler(y[1]) != sozcukler(y[2])]
    v = vurgu_sinavi(lambda w, ozel: vurgulu_hece(kucuk(w), ozel), yaz=False)
    lit = [v["literatur"][t] for t in ("kural", "sezer") if t in v["literatur"]]
    o = okunus_sinavi(sozcuk_okunusu, yaz=False)
    sonuc = {"hazirlama_fark": len(fark), "hazirlama_ornek": n["ornek"],
             "vurgu_literatur": sum(d for d, _ in lit) / sum(t for _, t in lit),
             "vurgu_kesin": v["vikisozluk_kesin"][0] / v["vikisozluk_kesin"][1], "okunus_per": o["per"]}
    if yaz:
        print(f"metin hazırlama  {n['ornek'] - len(fark)} / {n['ornek']} (harf ve noktalama farkı sayılmadan)")
        for girdi, cikti, beklenen in fark:
            print(f"    {girdi!r} → {cikti!r} (beklenen {beklenen!r})")
        print(f"vurgu            literatür kural+Sezer %{100 * sonuc['vurgu_literatur']:.1f} · Vikisözlük kesin "
              f"%{100 * sonuc['vurgu_kesin']:.1f}")
        print(f"okunuş           PER %{100 * sonuc['okunus_per']:.2f} ({o['sozcuk']} sözcük)")
    return sonuc


if __name__ == "__main__":
    s = olc()
    if "--esik" in sys.argv:
        dusen = [k for k in ESIK if (s[k] > ESIK[k] if k in ("hazirlama_fark", "okunus_per") else s[k] < ESIK[k])]
        if dusen:
            print("EŞİK ALTINDA: " + ", ".join(dusen))
            sys.exit(1)
        print("eşikler tamam")
