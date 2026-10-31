# -*- coding: utf-8 -*-
"""
VURGULU EĞİTİM LİSTELERİ — veri/egitim/ listelerinin aynısı, okunuş vurgu işaretli (ˈ).

Ses dosyaları ve train/val ayrımı DEĞİŞMEZ (makinedeki wavs/ yeniden gönderilmez);
yalnız metin sütunu g2p(metin, vurgu=True) olur. Çıktı: veri/egitim_vurgu/
  train_list.txt, val_list.txt, OOD_texts.txt ("okunuş|0": ayraçsız satırda satır sonu
  metne karışıyordu)

    python ogrenci/veri/vurguListe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "ogrenci"))
from g2p import g2p  # noqa: E402

KAYNAK, HEDEF = KOK / "veri" / "egitim", KOK / "veri" / "egitim_vurgu"


def main() -> int:
    metin: dict[str, str] = {}
    for ses in json.loads((KAYNAK / "konusmacilar.json").read_text(encoding="utf-8")):
        for s in (KOK / "cikti" / "veri" / ses / "olcum.jsonl").read_text(encoding="utf-8").splitlines():
            o = json.loads(s)
            metin[f"{ses}/{o['kimlik']}.flac"] = o["metin"]         # veriHazirla gibi: sonuncusu
    HEDEF.mkdir(parents=True, exist_ok=True)
    for ad in ("train_list.txt", "val_list.txt"):
        satir = []
        for s in (KAYNAK / ad).read_text(encoding="utf-8").splitlines():
            dosya, _, spk = s.split("|")
            satir.append(f"{dosya}|{g2p(metin[dosya], vurgu=True)}|{spk}")
        (HEDEF / ad).write_text("\n".join(satir) + "\n", encoding="utf-8")
        print(ad, len(satir), "·", satir[0])
    okunan = set(metin.values())
    korpus = [json.loads(s)["metin"] for s in (KOK / "veri" / "korpus.jsonl").read_text(encoding="utf-8").splitlines()]
    ood = [g2p(m, vurgu=True) + "|0" for m in korpus if m not in okunan and len(m) > 25]
    (HEDEF / "OOD_texts.txt").write_text("\n".join(ood) + "\n", encoding="utf-8")
    print("OOD", len(ood), "·", ood[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
