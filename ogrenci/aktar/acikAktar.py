# -*- coding: utf-8 -*-
"""
AÇIK MODEL AKTARIMI — seslen-tts (açık kaynak, PyTorch'suz) için model klasörünü hazırlar.

Seslen'de metin tarafı PyTorch'ta, çözücü ONNX'te (onnxAktar.py). Açık pakette ikisi de ONNX olsun diye metin
tarafı (metin kodlayıcı + Türkçe PL-BERT + süre kodlayıcı + süre tahmini) tek ağa toplanır. StyleTTS2'nin
pack_padded_sequence / numpy uzunluk dönüşümleri ONNX'e aktarılamıyor; tek cümlede (batch 1, dolgusuz) etkisiz
oldukları için eşdeğeri düz yazılır ve Seslen'in kendi yoluyla (model.Ogrenci.hazirla) karşılaştırılır.

Çıktı (cikti/acik/seslen-tts/):
  metin.onnx      girdi tokens (1, L) int64, stil (1, 256) → d (1, L, C), t_en (1, 512, L), sure (L,)
  ses.onnx        çözücü (onnxAktar.py çıktısı, soru ezgisi girdisi "egri" ile)
  stiller.npz     yalnız AÇIK seslerin ses × ton stil vektörleri (ürün sesleri Damla/Kuzey ASLA girmez)
  sesler.json     ses adı, görünen ad, açıklama, tonlar
  semboller.json  okunuş karakteri → işaret numarası (StyleTTS2 TextCleaner)

    python ogrenci/aktar/acikAktar.py [ağırlık]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "ogretmen")]
ACIK_SESLER = ["ayse", "zeynep", "emre", "fatma", "hasan", "koca"]
YASAK = {"damla", "kuzey"}                     # ürün sesleri
CUMLE = ["Tamam.", "Bugün hava çok güzel, parka gidelim mi?",
         "Ankara'dan gelen ekip, yarın sabah saat dokuzda İzmir'e geçecek; rapor da o zaman tamamlanacak."]
HEDEF = KOK / "cikti" / "acik" / "seslen-tts"


class MetinAgi(torch.nn.Module):
    """model.Ogrenci.hazirla'nın PyTorch kısmı, batch 1 ve dolgusuz: paketleme yok, maske yok."""

    def __init__(self, m):
        super().__init__()
        self.te, self.bert, self.bert_encoder, self.pr = m.text_encoder, m.bert, m.bert_encoder, m.predictor

    def forward(self, tokens: torch.Tensor, stil: torch.Tensor):
        from models import AdaLayerNorm
        s = stil[:, 128:]
        # metin kodlayıcı
        x = self.te.embedding(tokens).transpose(1, 2)
        for c in self.te.cnn:
            x = c(x)
        x, _ = self.te.lstm(x.transpose(1, 2))
        t_en = x.transpose(-1, -2)
        # PL-BERT + süre kodlayıcı
        d_en = self.bert_encoder(self.bert(tokens, attention_mask=torch.ones_like(tokens))).transpose(-1, -2)
        de = self.pr.text_encoder
        x = d_en.permute(2, 0, 1)
        sx = s.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, sx], dim=-1).transpose(0, 1).transpose(-1, -2)
        for blok in de.lstms:
            if isinstance(blok, AdaLayerNorm):
                x = blok(x.transpose(-1, -2), s).transpose(-1, -2)
                x = torch.cat([x, sx.permute(1, 2, 0)], dim=1)
            else:
                x, _ = blok(x.transpose(-1, -2))
                x = x.transpose(-1, -2)
        d = x.transpose(-1, -2)
        y, _ = self.pr.lstm(d)
        sure = torch.sigmoid(self.pr.duration_proj(y)).sum(dim=-1).squeeze(0)
        return d, t_en, sure


def ozgun(o, metin: str, stil: np.ndarray):
    """Seslen'in yolu (model.Ogrenci.hazirla ile aynı adımlar), yuvarlamadan önceki süreyle."""
    from utils import length_to_mask
    m = o.model
    kod = o.temiz(o.g2p(metin, vurgu=o.vurgu))
    t = torch.LongTensor([[0, *kod, 0]])
    n = torch.LongTensor([t.shape[-1]])
    maske = length_to_mask(n)
    ref_s = torch.from_numpy(stil).float().unsqueeze(0)
    s = ref_s[:, 128:]
    t_en = m.text_encoder(t, n, maske)
    d_en = m.bert_encoder(m.bert(t, attention_mask=(~maske).int())).transpose(-1, -2)
    d = m.predictor.text_encoder(d_en, s, n, maske)
    x, _ = m.predictor.lstm(d)
    sure = torch.sigmoid(m.predictor.duration_proj(x)).sum(axis=-1).squeeze(0)
    return t, ref_s, (d, t_en, sure)


@torch.no_grad()
def main() -> int:
    import onnxruntime as ort
    from kadro import KADRO
    from model import Ogrenci
    from yukle import sureci_hazirla
    sureci_hazirla()
    assert not YASAK & set(ACIK_SESLER), "ürün sesi açık listeye girmez"
    agirlik = Path(sys.argv[1]) if len(sys.argv) > 1 else KOK / "cikti" / "ogrenci" / "t10-epoch_2nd_00000.pth"
    o = Ogrenci(agirlik, onnx=False)
    agi = MetinAgi(o.model).eval()
    HEDEF.mkdir(parents=True, exist_ok=True)

    # 1) düz yazılmış ağ = Seslen'in yolu
    ornek = None
    for c in CUMLE:
        t, stil, (d0, te0, su0) = ozgun(o, c, o.stil_dosyadan("zeynep", "neseli"))
        d1, te1, su1 = agi(t, stil)
        fark = max(float((a - b).abs().max()) for a, b in ((d0, d1), (te0, te1), (su0, su1)))
        print(f"  eşdeğerlik ({t.shape[1]} işaret): en büyük fark {fark:.2e}", flush=True)
        if fark > 1e-4:
            print("GEÇMEDİ")
            return 1
        ornek = (t, stil) if ornek is None or t.shape[1] > ornek[0].shape[1] else ornek

    # 2) ONNX'e aktar + ONNX Runtime = PyTorch
    yol = HEDEF / "metin.onnx"
    torch.onnx.export(agi, ornek, str(yol), input_names=["tokens", "stil"], output_names=["d", "t_en", "sure"],
                      opset_version=17, dynamo=False, do_constant_folding=True,
                      dynamic_axes={"tokens": {1: "L"}, "d": {1: "L"}, "t_en": {2: "L"}, "sure": {0: "L"}})
    oturum = ort.InferenceSession(str(yol), providers=["CPUExecutionProvider"])
    for c in CUMLE:
        t, stil, beklenen = ozgun(o, c, o.stil_dosyadan("hasan", "uyari"))
        cikti = oturum.run(None, {"tokens": t.numpy(), "stil": stil.numpy()})
        fark = max(float(np.abs(a.numpy() - b).max()) for a, b in zip(beklenen, cikti))
        print(f"  ONNX = PyTorch ({t.shape[1]} işaret): en büyük fark {fark:.2e}", flush=True)
        if fark > 1e-3:
            print("GEÇMEDİ")
            return 1

    # 3) çözücü, açık seslerin stilleri, ses bilgileri, sembol tablosu
    shutil.copy(KOK / "cikti" / "ogrenci" / f"{o.onek}ses.onnx", HEDEF / "ses.onnx")
    v = np.load(KOK / "cikti" / "ogrenci" / f"{o.onek}stiller.npz")
    stiller = {k: v[k] for k in v.files if k.split("/")[0] in ACIK_SESLER}
    assert not any(k.split("/")[0] in YASAK for k in stiller)
    np.savez(HEDEF / "stiller.npz", **stiller)
    from studyoSunucu import GORUNEN, _aciklama
    sesler = {s: {"gorunen": GORUNEN.get(s, s.capitalize()), "tur": KADRO[s]["tur"],
                  "aciklama": _aciklama(KADRO[s]["aciklama"]),
                  "tonlar": sorted(k.split("/")[1] for k in stiller if k.startswith(s + "/"))} for s in ACIK_SESLER}
    (HEDEF / "sesler.json").write_text(json.dumps(sesler, ensure_ascii=False, indent=1), encoding="utf-8")
    (HEDEF / "semboller.json").write_text(json.dumps(o.temiz.word_index_dictionary, ensure_ascii=False), encoding="utf-8")
    boy = sum(f.stat().st_size for f in HEDEF.iterdir()) / 1e6
    print(f"yazıldı: {HEDEF} ({len(stiller)} stil, {boy:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
