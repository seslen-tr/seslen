# -*- coding: utf-8 -*-
"""
ÜRETİM PLANI — her ses için hangi cümle, hangi duyguyla, hangi kiple okunacak.

Havuzlar:
  veri/korpus.jsonl          nötr (yalnız "." ile bitenler; CC0 + bizim cümleler)
  veri/duygu/<duygu>.txt     duyguya UYGUN yazılmış cümleler (neşeli, heyecan,
                             uyarı, merak, özür; 700'er) — anlam ile ton uyuşsun
  veri/duygu/j.txt           "j"li nötr cümleler (öğretmen j'de zorlanıyor)
Yeni ses (varsayılan 2000 cümle, ~1,8 sa): nötr 640 + j 60 + her duygudan 260.
Damla / Kuzey (nötr verileri Kaggle'dan var): her duygudan 200 + j 60.
Her ses havuzdan KENDİ alt kümesini alır (crc32(ses|metin) sırası): aynı cümle
birkaç seste olabilir, hepsinde olmaz — metin çeşitliliği artar.
Kip (ogretmen/kadro.py KIP_NOTR): duygulu → B, nötr → sese göre B/C.

    python bulut/kiralik/plan.py        → veri/plan/<ses>.jsonl + özet
    python bulut/kiralik/plan.py yeni   → veri/plan_yeni/<ses>.jsonl: yalnız 03.10.2026'da eklenen
                                          üzgün / kızgın / fısıltı, her sese her birinden 200
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "ogretmen"))
from kadro import KIP_NOTR, secilenler  # noqa: E402

DUYGULAR = ["neseli", "heyecan", "uyari", "merak", "ozur"]
ESKI = {"damla", "kuzey"}
YENI_PAY = {"notr": 640, "j": 60, **{d: 260 for d in DUYGULAR}}
ESKI_PAY = {"j": 60, **{d: 200 for d in DUYGULAR}}
YENI_DUYGULAR = ["uzgun", "kizgin", "fisilti"]
YENI_DUYGU_PAY = {d: 200 for d in YENI_DUYGULAR}
ORT_SN = 3.3        # derlemde ölçülen ortalama cümle süresi


def havuzlar() -> dict[str, list[tuple[str, str]]]:
    h: dict[str, list[tuple[str, str]]] = {}
    korpus = [json.loads(s) for s in (KOK / "veri" / "korpus.jsonl").read_text(encoding="utf-8").splitlines()]
    h["notr"] = [(o["kimlik"], o["metin"]) for o in korpus if o["metin"].endswith(".")]
    on = {"neseli": "ne", "heyecan": "he", "uyari": "uy", "merak": "me", "ozur": "oz", "j": "j",
          "uzgun": "uz", "kizgin": "ki", "fisilti": "fi"}
    for ad, k in on.items():
        satirlar = [s.strip() for s in (KOK / "veri" / "duygu" / f"{ad}.txt").read_text(encoding="utf-8").splitlines()]
        h[ad] = [(f"{k}{i:04d}", s) for i, s in enumerate(satirlar, 1) if s]
    return h


def plan(ses: str, h: dict, pay: dict | None = None) -> list[dict]:
    pay = pay or (ESKI_PAY if ses in ESKI else YENI_PAY)
    out = []
    for havuz, n in pay.items():
        sirali = sorted(h[havuz], key=lambda km: zlib.crc32(f"{ses}|{km[1]}".encode()))[:n]
        duygu = "notr" if havuz in ("notr", "j") else havuz
        kip = KIP_NOTR[ses] if duygu == "notr" else "B"
        out += [{"kimlik": k, "metin": m, "duygu": duygu, "kip": kip} for k, m in sirali]
    # sıra karışık: iş yarıda kesilse de her duygudan veri olsun
    return sorted(out, key=lambda o: zlib.crc32(f"{ses}|sira|{o['kimlik']}".encode()))


def main() -> int:
    h = havuzlar()
    yeni = sys.argv[1:] == ["yeni"]
    hedef = KOK / "veri" / ("plan_yeni" if yeni else "plan")
    hedef.mkdir(parents=True, exist_ok=True)
    toplam = 0
    for ses in secilenler():
        p = plan(ses, h, YENI_DUYGU_PAY if yeni else None)
        (hedef / f"{ses}.jsonl").write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in p),
                                            encoding="utf-8")
        toplam += len(p)
        print(f"{ses:9s} {len(p):5d} cümle  ~{len(p) * ORT_SN / 3600:.1f} sa  nötr kipi {KIP_NOTR[ses]}")
    print(f"TOPLAM {toplam} cümle, ~{toplam * ORT_SN / 3600:.1f} sa ses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
