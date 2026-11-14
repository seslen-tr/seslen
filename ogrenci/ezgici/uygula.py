# -*- coding: utf-8 -*-
"""
EZGİCİYİ UYGULAMA — model.konus süreleri kesinleştirip öğrencinin perdesini hesapladıktan sonra çağırır
(Ezgi.cagir); ezgici hece hece örnek çeker, sonuç işaret başına yarım ton olarak soru ezgisinin yoluna (egri) eklenir.
Çözücü aynı, ONNX'te de çalışır. Konuşurken çağrıldığı için özdenetimin onarımları (okunuş, süre) da kendiliğinden
doğru eşlenir.

Sıcaklık 0,7 (14.11.2026, kulak: "önerin iyi"): 30 görülmemiş cümlede benzerlik 0,929 → 0,942, UTMOS 3,10 → 3,17,
takılan aynı (ogrenci/ezgici/olc.py). Süre örneklemesi ölçüde bir şey kazandırmadı, kullanılmıyor.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "ogrenci" / "ezgici")]
AGIRLIK = KOK / "cikti" / "ezgici" / "ezgici.pt"
SICAKLIK = 0.7
SINIR = 9.0                 # toplam perde farkı en çok ± yarım ton (uç örnek sesi bozmasın)


def hece_egrisi(hece: list[dict], uzunluk: int, duzey: float, fark: np.ndarray) -> np.ndarray:
    """Düzey + hece başına 5 noktalı fark → işaret başına ek yarım ton (uzunluk: baştaki ve sondaki dolgu dahil).
    Sözcük arası ve noktalama düzeyi alır."""
    ek = np.full(uzunluk, duzey, np.float32)
    for h, c in zip(hece, fark):
        i = np.array(h["isaret"])
        ek[i] += np.interp(np.linspace(0, 1, len(i)), np.linspace(0, 1, len(c)), c).astype(np.float32)
    return np.clip(ek, -SINIR, SINIR)


class Ezgi:
    def __init__(self, yol: Path = AGIRLIK):
        from ag import yukle
        self.ag, b = yukle(yol)
        self.sesler, self.tonlar = b["sesler"], b["tonlar"]
        self._kilit = threading.Lock()

    def var_mi(self, ses: str) -> bool:
        return ses in self.sesler

    def ek(self, ses: str, ton: str, okunus: str, sure: np.ndarray, f0: np.ndarray, sozluk: dict,
           sicaklik: float = SICAKLIK) -> np.ndarray | None:
        """Öğrencinin işaret süreleri ve F0'ı (konus'taki) → işaret başına ek yarım ton; ses bilinmiyorsa ya da
        cümle tek heceyse None. Tohum okunuştan: aynı cümle hep aynı okunur."""
        from ag import BITIS, BOYUT_K, OLCEK, kosul_kur, ornekle, tohum
        from ezgi import bitis, hece_ezgisi, ozellik
        if ses not in self.sesler:
            return None
        e = hece_ezgisi(okunus, sure, f0, sozluk)
        if len(e["hece"]) < 2:
            return None
        k = np.concatenate([np.zeros((1, BOYUT_K), np.float32), kosul_kur(ozellik(e["hece"]), e["sure"], e["cizgi"])])
        tid = self.tonlar.index(ton) if ton in self.tonlar else self.tonlar.index("notr")
        kimlik = lambda v: torch.LongTensor([v])
        with self._kilit:
            y = ornekle(self.ag, torch.from_numpy(k).unsqueeze(0), kimlik(self.sesler.index(ses)), kimlik(tid),
                        kimlik(BITIS.index(bitis(okunus))), torch.zeros(1, len(k)).bool(), tohum(okunus),
                        sicaklik=sicaklik)[0] * OLCEK
        y = y.numpy()
        return hece_egrisi(e["hece"], len(sure), float(y[0, 0]), y[1:, :5])

    def cagir(self, ses: str, ton: str, sozluk: dict, sicaklik: float = SICAKLIK):
        """model.konus(ezgici=…) için: (okunus, sure, f0) → ek yarım ton ya da None."""
        if ses not in self.sesler:
            return None
        return lambda okunus, sure, f0: self.ek(ses, ton, okunus, sure, f0, sozluk, sicaklik)
