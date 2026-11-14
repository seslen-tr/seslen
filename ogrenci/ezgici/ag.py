# -*- coding: utf-8 -*-
"""
EZGİCİ AĞI — öğrencinin çizgisine öğretmenden örneklenmiş sapma ekleyen küçük akış eşleme (flow matching) modeli.

Öğrencinin perde tahmincisi ortalamaya eğitildiği için ezgisi düz: cümle düzeyinin saçılması öğretmenin yarısı.
Ortalamaya eğitilen ikinci bir ağ da bunu düzeltemez, çünkü öğretmenin cümleden cümleye düzey değiştirmesi büyük ölçüde
rastlantısal. Burada ağ tek bir doğru yanıt değil, öğretmenin yanıtlarının dağılımını öğrenir; her cümlede ondan
tohumu metne bağlı bir örnek çekilir.

Dizinin başında cümle işareti (düzey), ardından hece işaretleri (perde çizgisi 5 nokta + süre) vardır. Hedefler
yarım ton / log oran cinsinden öğrenciye göre farktır, yani ağ sıfır üretirse öğrencinin ezgisi aynen kalır.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import torch
from torch import nn

NOKTA = 5
BOYUT_Y = NOKTA + 1         # perde farkı + süre farkı; cümle işaretinde ilk boyut düzey farkı
BOYUT_K = 14                # hece koşulu: banka özelliği 5, öğrenci çizgisi 5, ölçüsü 1, log süre 1, hece sayısı 1, sesli 1
OLCEK = torch.tensor([3.0] + [2.5] * (NOKTA - 1) + [0.4])       # hedefleri ~1 sapmaya getirir (ilk boyut düzey için de)
BITIS = ".?!"


class Ezgici(nn.Module):
    def __init__(self, ses_sayisi: int, ton_sayisi: int, d: int = 128, katman: int = 4, kafa: int = 4):
        super().__init__()
        self.ayar = {"ses_sayisi": ses_sayisi, "ton_sayisi": ton_sayisi, "d": d, "katman": katman, "kafa": kafa}
        self.giris = nn.Linear(BOYUT_Y + BOYUT_K, d)
        self.ses = nn.Embedding(ses_sayisi, d)
        self.ton = nn.Embedding(ton_sayisi, d)
        self.bitis = nn.Embedding(len(BITIS), d)
        self.cumle = nn.Parameter(torch.zeros(d))
        self.zaman = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.konum = nn.Linear(1, d)
        kat = nn.TransformerEncoderLayer(d, kafa, d * 3, dropout=0.1, activation="gelu", batch_first=True,
                                         norm_first=True)
        self.govde = nn.TransformerEncoder(kat, katman, enable_nested_tensor=False)
        self.son = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, BOYUT_Y))
        nn.init.zeros_(self.son[1].weight)
        nn.init.zeros_(self.son[1].bias)

    def _zaman(self, t: torch.Tensor) -> torch.Tensor:
        d = self.ayar["d"]
        f = torch.exp(torch.arange(d // 2, device=t.device) * (-math.log(1000.0) / (d // 2 - 1)))
        return self.zaman(torch.cat([torch.sin(t[:, None] * 1000 * f), torch.cos(t[:, None] * 1000 * f)], -1))

    def forward(self, x, t, kosul, ses, ton, bitis, bos):
        """x (B, 1+N, BOYUT_Y) gürültülü hedef; kosul (B, 1+N, BOYUT_K); bos (B, 1+N): dolgu → (B, 1+N, BOYUT_Y) hız."""
        h = self.giris(torch.cat([x, kosul], -1))
        h = h + self.konum(kosul[..., 4:5])
        h = torch.cat([h[:, :1] + self.cumle, h[:, 1:]], 1)        # yerinde değil: ONNX'e aktarılabilsin
        h = h + (self._zaman(t) + self.ses(ses) + self.ton(ton) + self.bitis(bitis)).unsqueeze(1)
        return self.son(self.govde(h, src_key_padding_mask=bos))


def akis_kaybi(ag: Ezgici, y, kosul, ses, ton, bitis, bos, maske) -> torch.Tensor:
    """Doğrusal yol üzerinde hız kaybı. maske (B, 1+N, BOYUT_Y): hedefi bilinmeyen boyutlar (sessiz hece) 0."""
    z = torch.randn_like(y)
    t = torch.rand(len(y), device=y.device)
    t_ = t[:, None, None]
    v = ag((1 - t_) * z + t_ * y, t, kosul, ses, ton, bitis, bos)
    return (((v - (y - z)) ** 2) * maske).sum() / maske.sum().clamp(min=1)


@torch.no_grad()
def ornekle(ag: Ezgici, kosul, ses, ton, bitis, bos, tohum: int, adim: int = 8, sicaklik: float = 1.0) -> torch.Tensor:
    """Gürültüden Euler ile hedef örneği (B, 1+N, BOYUT_Y), ölçeklenmiş. sicaklik < 1 ortalamaya yaklaştırır."""
    g = torch.Generator().manual_seed(tohum & 0x7FFFFFFF)
    x = torch.randn((*kosul.shape[:2], BOYUT_Y), generator=g) * sicaklik
    for i in range(adim):
        t = torch.full((len(x),), i / adim)
        x = x + ag(x, t, kosul, ses, ton, bitis, bos) / adim
    return x


def tohum(okunus: str, ek: int = 0) -> int:
    return zlib.crc32(okunus.encode()) + ek


def kosul_kur(oz: np.ndarray, ogr_sure: np.ndarray, ogr_cizgi: np.ndarray) -> np.ndarray:
    """Hece özellikleri → (N, BOYUT_K). Öğrenci çizgisi yarım ton; sessiz hecede 0 ve "sesli" = 0."""
    sesli = ~np.isnan(ogr_cizgi).any(1, keepdims=True)
    c = np.clip(np.nan_to_num(ogr_cizgi, nan=0.0), -14, 14)     # uçlar perde çıkarıcının hatası
    sure = np.log(np.maximum(ogr_sure, 1))[:, None]
    n = np.full((len(oz), 1), np.log(len(oz)) / 4, np.float32)
    return np.concatenate([oz, c / 4, np.abs(c).mean(1, keepdims=True) / 4, sure / 3, n, sesli], 1).astype(np.float32)


def yukle(yol) -> tuple[Ezgici, dict]:
    v = torch.load(yol, map_location="cpu", weights_only=True)
    ag = Ezgici(**v["ayar"])
    ag.load_state_dict(v["agirlik"])
    return ag.eval(), {"sesler": v["sesler"], "tonlar": v["tonlar"]}
