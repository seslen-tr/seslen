# -*- coding: utf-8 -*-
"""
EZGİCİ VERİSİ — ezgi bankasının (cikti/ezgi) bilmediği iki şey, her öğretmen kaydı için:
  orta       öğretmenin cümle düzeyi (ortanca F0, Hz). Banka perdeyi bu düzeye göre saklar, düzeyin kendisini atar;
             oysa öğrencinin öğretmenden en büyük farkı bu (README, "Öğretmenden fark nerede?").
  ogr_*      öğrencinin aynı cümle için ürün yoluyla ürettiği düzey, hece süreleri ve perde çizgisi (koşul).
Sonuç cikti/ezgici/veri/<ses>.npz; yarım kalırsa kalan sesten sürer. Aynı anda çok işçi: --parca i/n.

    python ogrenci/ezgici/veri.py [--parca 0/4] [--adet 50] [ses …]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "ogrenci" / "ezgici"), str(KOK / "ogretmen")]
CIKTI = KOK / "cikti" / "ezgici" / "veri"
BANKA = KOK / "cikti" / "ezgi"


def ortanca(f0: np.ndarray) -> float:
    s = f0[f0 > 40]
    return float(np.median(s)) if len(s) else 0.0


def ogrenci_ezgisi(ogr, okunus: str, stil: np.ndarray, sozluk: dict) -> tuple[float, np.ndarray, np.ndarray]:
    """Öğrencinin ürün yolundaki perdesi → (düzey Hz, hece süreleri, perde çizgisi: düzeye göre yarım ton)."""
    import torch
    from ezgi import hece_ezgisi
    from hizli import egri_carpani
    d, _, hiza, s, _, egri = ogr.hazirla("", stil, okunus=okunus, duraklar=[])
    with torch.no_grad():
        f0 = ogr.model.predictor.F0Ntrain(d.transpose(-1, -2) @ hiza, s)[0] * egri_carpani(egri, hiza)
    f0 = f0[0].numpy()
    e = hece_ezgisi(okunus, hiza[0].sum(-1).long().numpy(), f0, sozluk)
    return ortanca(f0), e["sure"], e["cizgi"]


def calis(sesler: list[str], parca: tuple[int, int], adet: int | None) -> None:
    import soundfile as sf
    import torch
    from ezgi import WAVS, mel
    from model import Ogrenci
    from studyo import URUN_AGIRLIK
    from yukle import sureci_hazirla
    from veriUret import uyanik_tut
    sureci_hazirla()
    uyanik_tut()
    ogr = Ogrenci(KOK / "cikti" / "ogrenci" / URUN_AGIRLIK, onnx=False)
    ogr.is_parcacigi = max(1, ogr.is_parcacigi // max(1, parca[1]))
    sozluk = ogr.temiz.word_index_dictionary
    CIKTI.mkdir(parents=True, exist_ok=True)
    i, n = parca
    for ses in sesler:
        hedef = CIKTI / f"{ses}.{i}.npz"
        if (CIKTI / f"{ses}.npz").exists() or hedef.exists():
            continue
        meta = json.loads((BANKA / f"{ses}.json").read_text(encoding="utf-8"))
        ofset = np.load(BANKA / f"{ses}.npz")["ofset"]
        sira = list(range(i, len(meta), n))[:adet]
        orta, ogr_orta, ogr_ofset, ogr_sure, ogr_cizgi, gecen, t0 = [], [], [0], [], [], [], time.perf_counter()
        for k, j in enumerate(sira):
            m = meta[j]
            x, _ = sf.read(WAVS / m["dosya"], dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            me = mel(ogr, x)
            me = me[:, : me.shape[1] // 2 * 2].unsqueeze(0)
            with torch.no_grad():
                f0 = ogr.model.pitch_extractor(me.unsqueeze(1))[0].numpy().reshape(-1)
            ogr.stil_dosyadan(ses)                          # stiller ilk çağrıda yüklenir
            ton = m["ton"] if f"{ses}/{m['ton']}" in ogr.stiller else "notr"
            stil = ogr.cumle_stili(m["okunus"], ogr.stil_dosyadan(ses, ton), okunus=m["okunus"])
            o, s, c = ogrenci_ezgisi(ogr, m["okunus"], stil, sozluk)
            if len(s) != ofset[j + 1] - ofset[j]:           # bankanın hece sayısı tutmazsa kayıt atlanır
                continue
            gecen.append(j)
            orta.append(ortanca(f0))
            ogr_orta.append(o)
            ogr_sure.append(s)
            ogr_cizgi.append(c)
            ogr_ofset.append(ogr_ofset[-1] + len(s))
            if k % 100 == 99:
                print(f"{ses}.{i}: {k + 1}/{len(sira)}, {(time.perf_counter() - t0) / (k + 1):.2f} sn/cümle", flush=True)
        np.savez_compressed(hedef, sira=np.array(gecen, np.int64), orta=np.array(orta, np.float32),
                            ogr_orta=np.array(ogr_orta, np.float32), ogr_ofset=np.array(ogr_ofset, np.int64),
                            ogr_sure=np.concatenate(ogr_sure), ogr_cizgi=np.concatenate(ogr_cizgi))
        print(f"{ses}.{i}: {len(gecen)} cümle, {time.perf_counter() - t0:.0f} sn", flush=True)


def main(argv: list[str]) -> None:
    parca, adet = (0, 1), None
    if "--parca" in argv:
        k = argv.index("--parca")
        parca = tuple(int(v) for v in argv[k + 1].split("/"))
        del argv[k:k + 2]
    if "--adet" in argv:
        k = argv.index("--adet")
        adet = int(argv[k + 1])
        del argv[k:k + 2]
    calis(argv or sorted(p.stem for p in BANKA.glob("*.npz")), parca, adet)


if __name__ == "__main__":
    main(sys.argv[1:])
