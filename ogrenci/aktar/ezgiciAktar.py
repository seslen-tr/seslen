# -*- coding: utf-8 -*-
"""
EZGİCİ AKTARIMI — seslen-tts (PyTorch'suz) için ezgiciyi ONNX'e aktarır, açık model klasörüne yazar:
  perde.onnx    öğrencinin perde tahmini: d (1, L, C), hiza (1, L, F), s (1, 128), egri (1, L) → f0 (1, 2F).
                ses.onnx'in içinde de var, ama ezgici çözücüden önce bu perdeye bakar (öğrencinin çizgisi koşuldur).
  ezgici.onnx   ezgicinin hız ağı, tek cümle, dolgusuz: x (1, N, 6), t (1,), kosul (1, N, 14), ses, ton, bitis (1,)
                → v (1, N, 6). Euler adımları pakette numpy ile atılır.
  ezgici.json   ses ve ton sırası (açık adlarla), ölçek, sıcaklık, adım, sınır.
Ürün seslerinin yerel adları dosyalara girmez (acikAktar.YASAK). ONNX çıktısı PyTorch'la karşılaştırılır.

    python ogrenci/aktar/ezgiciAktar.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "ogrenci" / "ezgici"), str(KOK / "ogrenci" / "aktar"),
                str(KOK / "ogretmen")]


class PerdeAgi(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.predictor = model.predictor

    def forward(self, d, hiza, s, egri):
        from hizli import egri_carpani
        f0, _ = self.predictor.F0Ntrain(d.transpose(-1, -2) @ hiza, s)
        return f0 * egri_carpani(egri, hiza)


class HizAgi(torch.nn.Module):
    def __init__(self, ag):
        super().__init__()
        self.ag = ag

    def forward(self, x, t, kosul, ses, ton, bitis):
        return self.ag(x, t, kosul, ses, ton, bitis, None)


def ust_veri_temizle(yol: Path) -> None:
    """dynamo aktarıcısı her düğüme kaynak yolunu ve yığın izini yazıyor (yerel klasör adları); açık dosyaya girmez."""
    import onnx
    m = onnx.load(str(yol))

    def temizle(dugumler):
        for d in dugumler:
            d.doc_string = ""
            del d.metadata_props[:]
            for a in d.attribute:
                if a.g is not None:
                    temizle(a.g.node)
    temizle(m.graph.node)
    for f in m.functions:
        del f.metadata_props[:]
        temizle(f.node)
    for t in [*m.graph.initializer, *m.graph.input, *m.graph.output, *m.graph.value_info]:
        t.doc_string = ""
        del t.metadata_props[:]
    m.graph.doc_string, m.doc_string = "", ""
    del m.metadata_props[:], m.graph.metadata_props[:]
    onnx.save(m, str(yol))
    with open(yol, "rb") as f:
        assert not re.search(rb"(?i)users[\/]|desktop|site-packages", f.read()), "ONNX'te yerel yol kaldı"


def esit(oturum, girdi: dict, beklenen: torch.Tensor, ad: str, esik: float) -> None:
    fark = float(np.abs(oturum.run(None, {k: v.numpy() for k, v in girdi.items()})[0] - beklenen.numpy()).max())
    print(f"  {ad}: ONNX = PyTorch, en büyük fark {fark:.2e}", flush=True)
    if fark > esik:
        raise SystemExit(f"{ad} GEÇMEDİ")


@torch.no_grad()
def main() -> None:
    import onnxruntime as ort
    from acikAktar import ACIK_AD, HEDEF, YASAK
    from ag import BOYUT_K, BOYUT_Y, OLCEK, yukle
    from model import Ogrenci
    from studyo import URUN_AGIRLIK
    from uygula import AGIRLIK, SICAKLIK, SINIR
    torch.backends.mha.set_fastpath_enabled(False)
    o = Ogrenci(KOK / "cikti" / "ogrenci" / URUN_AGIRLIK, onnx=False)
    HEDEF.mkdir(parents=True, exist_ok=True)

    # 1) perde.onnx
    perde = PerdeAgi(o.model).eval()
    ornekler = []
    for metin, ses in (("Ankara'dan gelen ekip yarın sabah İzmir'e geçecek mi?", "zeynep"), ("Tamam, geldim.", "hasan")):
        d, _, hiza, s, _, egri = o.hazirla(metin, o.stil_dosyadan(ses), duraklar=[])
        ornekler.append({"d": d, "hiza": hiza, "s": s, "egri": egri})
    yol = HEDEF / "perde.onnx"
    torch.onnx.export(perde, tuple(ornekler[0].values()), str(yol), input_names=list(ornekler[0]), output_names=["f0"],
                      opset_version=17, dynamo=False, do_constant_folding=True,
                      dynamic_axes={"d": {1: "L"}, "hiza": {1: "L", 2: "F"}, "egri": {1: "L"}, "f0": {1: "F2"}})
    ust_veri_temizle(yol)
    oturum = ort.InferenceSession(str(yol), providers=["CPUExecutionProvider"])
    for g in ornekler:
        esit(oturum, g, perde(**g), "perde", 1e-2)

    # 2) ezgici.onnx
    ag, bilgi = yukle(AGIRLIK)
    # ses sırası açık adlara göre: yerel adların sırası kalırsa açık adın yeri yerel adın baş harfini ele verir
    sesler = sorted(ACIK_AD.get(s, s) for s in bilgi["sesler"])
    sira = [[ACIK_AD.get(s, s) for s in bilgi["sesler"]].index(a) for a in sesler]
    ag.ses.weight.data = ag.ses.weight.data[sira].clone()
    hiz = HizAgi(ag).eval()
    rs = torch.Generator().manual_seed(0)
    ornekler = []
    for n in (24, 7):
        ornekler.append({"x": torch.randn(1, n, BOYUT_Y, generator=rs), "t": torch.tensor([0.375]),
                         "kosul": torch.randn(1, n, BOYUT_K, generator=rs), "ses": torch.LongTensor([3]),
                         "ton": torch.LongTensor([5]), "bitis": torch.LongTensor([1])})
    yol = HEDEF / "ezgici.onnx"
    # eski izleyici (dynamo=False) dikkat katmanında cümle uzunluğunu sabit gömüyordu; dynamo değişken tutuyor
    n = torch.export.Dim("N", min=2, max=512)
    torch.onnx.export(hiz, tuple(ornekler[0].values()), str(yol), input_names=list(ornekler[0]), output_names=["v"],
                      opset_version=18, dynamo=True, external_data=False,
                      dynamic_shapes={"x": {1: n}, "t": None, "kosul": {1: n}, "ses": None, "ton": None, "bitis": None})
    ust_veri_temizle(yol)
    oturum = ort.InferenceSession(str(yol), providers=["CPUExecutionProvider"])
    for g in ornekler:
        esit(oturum, g, hiz(**g), "ezgici", 1e-4)

    # 3) ezgici.json (açık adlarla)
    ayar = {"sesler": sesler, "tonlar": bilgi["tonlar"], "olcek": OLCEK.tolist(), "sicaklik": SICAKLIK, "adim": 8,
            "sinir": SINIR}
    metin = json.dumps(ayar, ensure_ascii=False)
    assert not YASAK & set(re.findall(r"\w+", metin.lower())), "yerel ad sızdı"
    (HEDEF / "ezgici.json").write_text(metin, encoding="utf-8")
    print(f"yazıldı: {HEDEF} (perde.onnx, ezgici.onnx, ezgici.json; {len(sesler)} ses)")


if __name__ == "__main__":
    main()
