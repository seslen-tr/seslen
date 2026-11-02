# -*- coding: utf-8 -*-
"""
ÇÖZÜCÜYÜ ONNX'E AKTAR + DOĞRULA — hız kazancı sesi DEĞİŞTİRMEMELİ.

    python ogrenci/aktar/onnxAktar.py cikti/ogrenci/epoch_2nd_00000.pth

Üç sınav (hepsi geçmezse dosya yazılmaz):
  1) Yamalı PyTorch (hizli.py) = özgün PyTorch, aynı tohumla       → SNR ≥ 50 dB
  2) ONNX Runtime = yamalı PyTorch, iki tarafta aynı sabit gürültü: log-mel farkı, modelin KENDİ
     iki çalıştırması (farklı gürültü tohumu) arasındaki farktan büyük olmamalı. Dalga SNR'si burada
     yanıltır: uzun cümlede harmonik kaynağın fazı float32 birikimli toplamda yavaşça kayıyor
     (01.10: 40 / 36 / 25 dB), tayf aynı kalıyor.
  50 dB: fark sinyal genliğinin ~%0,3'ü, duyulmaz. Tam eşitlik beklenemez: neredeyse gerçel
  STFT kutularında torch'un fazı da yuvarlama gürültüsüyle +π/−π seçiyor (01.10: 58,7 dB).
  3) Süre: özgün PyTorch'a karşı ONNX Runtime, üç cümle boyunda
Çıktı: cikti/ogrenci/<önek>ses.onnx (gürültülü, üretim; 3. aşama: t3-ses.onnx) — metin tarafı PyTorch'ta
(sürenin ~%5'i). model.Ogrenci bu dosyayı varsa kendiliğinden kullanır.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "ogrenci"))
sys.path.insert(0, str(KOK / "ogretmen"))
from hizli import SesAgi, yamala  # noqa: E402
from model import Ogrenci  # noqa: E402

CUMLE = ["Tamam.",
         "Merhaba, ben buradayım. Bugün haritada birlikte çalışalım mı?",
         "Kuzey mahallesindeki yeni su hattı ölçümleri rapora eklendi; ayrıntıları yarın sabah "
         "birlikte inceleyip eksik parselleri tek tek işaretleyeceğiz, sonra da belediyeye göndereceğiz."]
GIRDI = ["d", "t_en", "hiza", "s", "ref", "perde", "tonlama", "egri"]


def snr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return float(10 * np.log10((a ** 2).sum() / max(((a - b) ** 2).sum(), 1e-20)))


def mel_fark(a: np.ndarray, b: np.ndarray) -> float:
    """İki dalganın log-mel tayfı arasındaki ortalama mutlak fark (dB) — fazdan bağımsız, kulağa yakın."""
    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=2048, win_length=1200, hop_length=300, n_mels=80)
    n = min(len(a), len(b))
    la, lb = (10 * torch.log10(mel(torch.from_numpy(x[:n].copy())) + 1e-5) for x in (a, b))
    return float((la - lb).abs().mean())


def girdiler(o: Ogrenci, metin: str, stil) -> tuple:
    d, t_en, hiza, s, ref, egri = o.hazirla(metin, stil)
    return d, t_en, hiza, s, ref, torch.ones(1), torch.ones(1), egri


def aktar(agi: SesAgi, ornek: tuple, yol: Path) -> None:
    torch.onnx.export(agi, ornek, str(yol), input_names=GIRDI, output_names=["ses"], opset_version=17,
                      dynamo=False, do_constant_folding=True,
                      dynamic_axes={"d": {1: "L"}, "t_en": {2: "L"}, "hiza": {1: "L", 2: "F"}, "egri": {1: "L"},
                                    "ses": {0: "T"}})


class _Sessiz:
    """rand_like/randn_like → iki tarafta da AYNI sabit gürültü (tampondan kesit), gürültü dışında
    her şey karşılaştırılabilsin. Sıfır gürültü olmaz: sessiz yerlerde genlik ~0 kalır, faz tanımsızlaşır."""

    def __enter__(self):
        g = torch.Generator().manual_seed(0)
        tu, tn = torch.rand(3_000_000, generator=g), torch.randn(3_000_000, generator=g)
        self.r, self.n = torch.rand_like, torch.randn_like
        torch.rand_like = lambda x, **k: tu[: x.flatten().shape[0]].reshape(x.shape)
        torch.randn_like = lambda x, **k: tn[: x.flatten().shape[0]].reshape(x.shape)

    def __exit__(self, *a):
        torch.rand_like, torch.randn_like = self.r, self.n


def main() -> int:
    import onnxruntime as ort
    from yukle import sureci_hazirla
    sureci_hazirla()
    o = Ogrenci(sys.argv[1], onnx=False)
    stil = o.stil_dosyadan("zeynep", "neseli")
    agi = SesAgi(o.model).eval()
    ornekler = [girdiler(o, c, stil) for c in CUMLE]

    with torch.no_grad():
        torch.manual_seed(0)
        ozgun = [agi(*g).numpy() for g in ornekler]
        sure_ozgun = []
        for g in ornekler:
            t = time.perf_counter()
            agi(*g)
            sure_ozgun.append(time.perf_counter() - t)
        yamala(o.model, konv2b=os.environ.get("KONV2B", "0") == "1")
        torch.manual_seed(0)
        yamali = [agi(*g).numpy() for g in ornekler]
    s1 = min(snr(a, b) for a, b in zip(ozgun, yamali))
    print(f"1) yamalı = özgün: en kötü SNR {s1:.1f} dB", flush=True)
    if s1 < 50:
        print("   GEÇMEDİ")
        return 1

    hedef = KOK / "cikti" / "ogrenci"
    deneme = hedef / "ses-sessiz.onnx"
    with torch.no_grad(), _Sessiz():
        sessiz = [agi(*g).numpy() for g in ornekler]
        aktar(agi, ornekler[1], deneme)
    secenek = ort.SessionOptions()
    secenek.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    oturum = ort.InferenceSession(str(deneme), secenek, providers=["CPUExecutionProvider"])
    ver = lambda g: {a: t.numpy() for a, t in zip(GIRDI, g)}
    with torch.no_grad():
        torch.manual_seed(1)
        baska = [agi(*g).numpy() for g in ornekler]          # aynı model, başka gürültü tohumu
    taban = [mel_fark(a, b) for a, b in zip(yamali, baska)]
    onnx_ = [mel_fark(a, oturum.run(None, ver(g))[0]) for a, g in zip(sessiz, ornekler)]
    print("2) log-mel farkı (dB) — ONNX:", " / ".join(f"{x:.3f}" for x in onnx_),
          "· modelin kendi iki çalıştırması:", " / ".join(f"{x:.3f}" for x in taban), flush=True)
    deneme.unlink()
    if any(x > t for x, t in zip(onnx_, taban)):
        print("   GEÇMEDİ")
        return 1

    with torch.no_grad():
        aktar(agi, ornekler[1], hedef / f"{o.onek}ses.onnx")
    oturum = ort.InferenceSession(str(hedef / f"{o.onek}ses.onnx"), secenek, providers=["CPUExecutionProvider"])
    for g in ornekler:
        oturum.run(None, ver(g))                     # ısınma
    print("3) süre (çözücü):")
    for c, g, so in zip(CUMLE, ornekler, sure_ozgun):
        t = time.perf_counter()
        w = oturum.run(None, ver(g))[0]
        so2 = time.perf_counter() - t
        print(f"   {len(w) / 24000:4.1f} sn ses: PyTorch {so * 1000:5.0f} ms → ONNX {so2 * 1000:5.0f} ms "
              f"(×{so / so2:.1f})", flush=True)
    yol = hedef / f"{o.onek}ses.onnx"
    print(f"yazıldı: {yol} ({yol.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
