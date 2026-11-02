# -*- coding: utf-8 -*-
"""
HIZLI ÇALIŞMA — çözücüyü ONNX'e aktarılabilir yapan, MATEMATİĞİ AYNI yamalar.

İşlemcide sürenin %91'i çözücüde (iSTFTNet; ogrenci/aktar/onnxAktar.py ölçer). ONNX
Runtime'a taşımak için iki parça değişir, sonuç değişmez:
  1) torch.stft / torch.istft (pencere 20, adım 5) → sabit DFT çekirdekli evrişim
     (Kokoro'nun ONNX sürümündeki yol). Kayan nokta farkı dışında birebir.
  2) SineGen._f02sine: yerinde dilim atamaları ve dinamik boyutlu torch.rand →
     rand_like + birleştirme. Aynı tohumla aynı rastgele sayıları çeker.
Ağırlık normalizasyonu da katlanır (her çağrıda yeniden hesaplanmasın).
"""
from __future__ import annotations

import math
import types

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import get_window
from torch import nn


class KonvSTFT(nn.Module):
    """torch.stft/istft(center=True, reflect, hann, onesided) ile aynı; yalnız conv1d."""

    def __init__(self, n_fft: int = 20, hop: int = 5):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        w = torch.from_numpy(get_window("hann", n_fft, fftbins=True).astype(np.float64))
        n = torch.arange(n_fft, dtype=torch.float64)
        f = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        aci = 2 * math.pi * f[:, None] * n[None, :] / n_fft
        c = torch.full((len(f),), 2.0, dtype=torch.float64)
        c[0] = 1.0
        c[-1] = 1.0                                      # n_fft çift: Nyquist bir kez
        k = lambda t: t[:, None, :].float()              # (frekans, 1, pencere)
        self.register_buffer("ileri_re", k(w * torch.cos(aci)))
        self.register_buffer("ileri_im", k(-w * torch.sin(aci)))
        self.register_buffer("geri_re", k(c[:, None] / n_fft * torch.cos(aci) * w))
        self.register_buffer("geri_im", k(-c[:, None] / n_fft * torch.sin(aci) * w))
        self.register_buffer("pencere2", (w ** 2)[None, None, :].float())
        # DC ve Nyquist kutusunun sanal kısmı tam 0 (FFT'de öyle); evrişimde ±1e-8 kalıyor ve atan2
        # fazı +π/−π arasında zıplatıyor. Ağ fazı sayı olarak okuduğundan torch.angle gibi 0/π verilir.
        kenar = torch.zeros(1, len(f), 1, dtype=torch.bool)
        kenar[0, 0, 0] = kenar[0, -1, 0] = True
        self.register_buffer("kenar", kenar)

    def transform(self, x: torch.Tensor):
        x = F.pad(x[:, None, :], (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        re = F.conv1d(x, self.ileri_re, stride=self.hop)
        im = F.conv1d(x, self.ileri_im, stride=self.hop)
        # İlk çerçeve de: yansıtmalı dolgu onu simetrik yapar → spektrum gerçel, faz yine 0/π.
        ilk = torch.arange(re.shape[-1]) == 0
        faz = torch.where(self.kenar | ilk, torch.where(re < 0, math.pi, 0.0), torch.atan2(im, re))
        return torch.sqrt(re * re + im * im), faz

    def inverse(self, genlik: torch.Tensor, faz: torch.Tensor) -> torch.Tensor:
        y = (F.conv_transpose1d(genlik * torch.cos(faz), self.geri_re, stride=self.hop)
             + F.conv_transpose1d(genlik * torch.sin(faz), self.geri_im, stride=self.hop))
        zarf = F.conv_transpose1d(torch.ones_like(genlik[:, :1, :]), self.pencere2, stride=self.hop)
        y = y / torch.clamp(zarf, min=1e-11)
        return y[..., self.n_fft // 2: -(self.n_fft // 2)]          # (B, 1, T)


class SolYansit(nn.Module):
    """ReflectionPad1d((1, 0)) ile aynı; ONNX Pad'den sonra kanal sayısı kayboluyor (InstanceNorm aktarılamıyor)."""

    def forward(self, x):
        return torch.cat([x[..., 1:2], x], dim=-1)


class Konv2B(nn.Module):
    """Conv1d'nin (1×k) Conv2d eşdeğeri: ONNX Runtime'ın hızlı NCHWc çekirdekleri yalnız 2B evrişimde."""

    def __init__(self, c: nn.Conv1d):
        super().__init__()
        self.c = nn.Conv2d(c.in_channels, c.out_channels, (1, c.kernel_size[0]), (1, c.stride[0]),
                           (0, c.padding[0]), (1, c.dilation[0]), c.groups, c.bias is not None)
        self.c.weight.data = c.weight.data.unsqueeze(2).clone()
        if c.bias is not None:
            self.c.bias.data = c.bias.data.clone()

    def forward(self, x):
        return self.c(x.unsqueeze(2)).squeeze(2)


def _snake_blok(self, x, s):
    """AdaINResBlock1.forward ile aynı; Snake x + sin²(αx)/α → üs alma yerine çarpma, 1/α önceden."""
    for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
        xt = n1(x, s)
        t = torch.sin(a1 * xt)
        xt = c1(xt + t * t * (1 / a1))
        xt = n2(xt, s)
        t = torch.sin(a2 * xt)
        x = c2(xt + t * t * (1 / a2)) + x
    return x


def _f02sine(self, f0_values: torch.Tensor) -> torch.Tensor:
    """Modules/istftnet.py SineGen._f02sine (flag_for_pulse=False) ile aynı hesap."""
    rad = (f0_values / self.sampling_rate) % 1
    ilk = torch.rand_like(f0_values[:, 0, :])                     # = torch.rand(B, dim)
    ilk = ilk * (torch.arange(ilk.shape[-1]) > 0).to(ilk.dtype)   # temel bileşene faz gürültüsü yok
    rad = torch.cat([rad[:, :1, :] + ilk[:, None, :], rad[:, 1:, :]], dim=1)
    rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear").transpose(1, 2)
    faz = torch.cumsum(rad, dim=1) * 2 * np.pi
    faz = F.interpolate(faz.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale,
                        mode="linear").transpose(1, 2)
    return torch.sin(faz)


def yamala(model, konv2b: bool = False) -> None:
    """Öğrenci modelini yerinde hızlı/aktarılabilir biçime getirir (çıkış aynı)."""
    from torch.nn.utils import remove_weight_norm
    for mod in model.decoder.modules():
        try:
            remove_weight_norm(mod)
        except (ValueError, AttributeError):
            pass
    g = model.decoder.generator
    g.stft = KonvSTFT(g.stft.filter_length, g.stft.hop_length)
    assert tuple(g.reflection_pad.padding) == (1, 0)
    g.reflection_pad = SolYansit()
    sg = g.m_source.l_sin_gen
    sg._f02sine = types.MethodType(_f02sine, sg)
    for blok in g.modules():
        if type(blok).__name__ == "AdaINResBlock1":
            blok.forward = types.MethodType(_snake_blok, blok)
    if konv2b:
        for ust in list(g.modules()):
            for ad, c in list(ust.named_children()):
                if isinstance(c, nn.Conv1d):
                    setattr(ust, ad, Konv2B(c))


def tonla(f0: torch.Tensor, perde: torch.Tensor, tonlama: torch.Tensor) -> torch.Tensor:
    """Stüdyo / üslup: perde = F0 çarpanı (ince-kalın); tonlama = sesli çerçevelerde ortalama perde
    etrafındaki sapmanın çarpanı (1 doğal, >1 hareketli, <1 düz). Sessiz çerçeveler (F0 ≈ 0) değişmez."""
    sesli = (f0 > 1.0).float()
    ort = (f0 * sesli).sum(dim=-1, keepdim=True) / sesli.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return (sesli * (ort + tonlama * (f0 - ort)).clamp(min=1.0) + (1.0 - sesli) * f0) * perde


def egri_carpani(egri: torch.Tensor, hiza: torch.Tensor) -> torch.Tensor:
    """İşaret başına yarım ton eki (1, L) → F0 çarpanı (1, 2F): hizalamayla çerçevelere yayılır, ~175 ms'lik
    pencereyle yumuşatılır (basamak değil, kayan bir tepe), F0 çözünürlüğüne (2× çerçeve) çıkarılır.
    Sessiz çerçevelerde F0 ≈ 0 olduğu için çarpan bir şey değiştirmez."""
    yt = torch.nn.functional.avg_pool1d((egri.unsqueeze(1) @ hiza), 7, stride=1, padding=3, count_include_pad=False)
    return torch.pow(2.0, torch.nn.functional.interpolate(yt, scale_factor=2.0, mode="nearest") / 12.0).squeeze(1)


class SesAgi(nn.Module):
    """Hizalanmış metin özellikleri + stil → dalga. Sürenin %95'i burada; ONNX'e giden kısım."""

    def __init__(self, model):
        super().__init__()
        self.predictor, self.decoder = model.predictor, model.decoder

    def forward(self, d, t_en, hiza, s, ref, perde, tonlama, egri):
        # d: (1, L, C) süre kodlayıcı çıkışı · t_en: (1, 512, L) · hiza: (1, L, F) · s/ref: (1, 128)
        # egri: (1, L) işaret başına yarım ton eki (soru ezgisi; model.soru_egrisi)
        en = d.transpose(-1, -2) @ hiza
        f0, n = self.predictor.F0Ntrain(en, s)
        f0 = f0 * egri_carpani(egri, hiza)
        return self.decoder(t_en @ hiza, tonla(f0, perde, tonlama), n, ref).reshape(-1)
