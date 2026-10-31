# -*- coding: utf-8 -*-
"""
Ses ölçüleri — adaylar aynı cetvelle karşılaştırılsın.

    netlik(x, sr)     CPP (kepstral tepe belirginliği, dB), sesli çerçevelerin
                      ortancası. Yüksek = net, düşük = hırıltılı. Başvuru
                      (seslen, 28.09.2026): Chatterbox öğretmen verisi 15-16,
                      eğitilmiş öğrenci 16-18, temizlik süzgeçli 25-26.
    dogruluk(x, sr, metin)
                      Whisper (medium, Türkçe) ne duydu, metinle sözcük
                      benzerliği (0-1). Telaffuz / yutulan hece ölçüsü.
    bant(x, sr)       sesin gerçekten dolu olduğu üst frekans (Hz): enerjinin
                      %99,5'i bu frekansın altında. 16 kHz kayıt ≤ 8000.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np

ORAN = 24000   # ölçüler ortak orana çevrilerek alınır


def _ortak(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != ORAN:
        import soxr
        x = soxr.resample(x, sr, ORAN).astype(np.float32)
    return x


def netlik(x: np.ndarray, sr: int) -> float:
    import librosa
    x = _ortak(x, sr)
    n, hop = 1024, 256
    _, sesli, _ = librosa.pyin(x, fmin=70, fmax=600, sr=ORAN, frame_length=n, hop_length=hop)
    F = librosa.util.frame(np.pad(x, n // 2), frame_length=n, hop_length=hop) * np.hanning(n)[:, None]
    L = 20 * np.log10(np.abs(np.fft.rfft(F, axis=0)) + 1e-9)
    C = 20 * np.log10(np.abs(np.fft.irfft(L, axis=0))[: n // 2] + 1e-9)
    q = np.arange(n // 2) / ORAN
    m = (q > 1 / 600) & (q < 1 / 70)
    out = []
    for j in np.where(sesli[: C.shape[1]])[0]:
        c, qq = C[m, j], q[m]
        a, b = np.polyfit(qq, c, 1)
        out.append(c.max() - (a * qq[c.argmax()] + b))
    return float(np.median(out)) if out else float("nan")


def bant(x: np.ndarray, sr: int) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    S = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    k = np.cumsum(S) / S.sum()
    return float(f[np.searchsorted(k, 0.995)])


@lru_cache(maxsize=1)
def _whisper():
    """Önce GPU: VoxCPM2 işlemcide çalışırken ekran kartı boş (RTX 3050, 4 GB;
    VoxCPM2 sığmıyor ama Whisper medium int8 ~1,5 GB sığar). Yoksa işlemci.
    CUDA kitaplıkları pip ile gelir (nvidia-cublas-cu12, nvidia-cudnn-cu12)."""
    import os
    import site
    from faster_whisper import WhisperModel
    for kok in site.getsitepackages():
        for alt in ("cublas", "cudnn"):
            d = os.path.join(kok, "nvidia", alt, "bin")
            if os.path.isdir(d):
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
    try:
        m = WhisperModel("medium", device="cuda", compute_type="int8_float16")
        m.transcribe(np.zeros(16000, dtype=np.float32), language="tr")   # DLL'ler gerçekten yükleniyor mu
        return m
    except Exception as e:
        print(f"Whisper GPU'da açılamadı ({type(e).__name__}: {e}); işlemcide", flush=True)
        return WhisperModel("medium", device="cpu", compute_type="int8")


_BIR = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_ON = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def _okunus(n: int) -> list[str]:
    """Tam sayının Türkçe okunuşu (sözcük listesi). Whisper '347' yazar, metin
    'üç yüz kırk yedi' der: ikisi aynı sayılsın."""
    if n == 0:
        return ["sıfır"]
    out: list[str] = []
    for bolen, ad in ((10 ** 9, "milyar"), (10 ** 6, "milyon"), (1000, "bin")):
        if n >= bolen:
            k, n = divmod(n, bolen)
            out += ([] if (k == 1 and ad == "bin") else _okunus(k)) + [ad]
    if n >= 100:
        k, n = divmod(n, 100)
        out += ([] if k == 1 else [_BIR[k]]) + ["yüz"]
    if n >= 10:
        out.append(_ON[n // 10])
        n %= 10
    if n:
        out.append(_BIR[n])
    return out


def _sade(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s.replace("I", "ı").replace("İ", "i").lower())
    # Şapka okunuşu değiştirmez ama Whisper çoğu zaman yazmaz ("zekâ" → "zeka").
    # ı/i: Whisper baştaki I'yı çoğu kez İ yazar ("Ihlamur" → "İhlamur").
    # Kesme işareti: "Zeytinlik'teki" tek sözcük (yoksa ikiye bölünüp ceza yiyordu).
    # Ondalık: Whisper "61,6" yazar, metin "altmış bir virgül altı" der.
    # Eşleme sayılar sözcüğe açıldıktan SONRA (okunuşta da "altı" var).
    s = re.sub(r"['’`]", "", s)
    out: list[str] = []
    for p in re.findall(r"[a-zçğıöşüâîû]+|\d+(?:,\d+)?", s):
        if p[0].isdigit():
            tam, _, kesir = p.partition(",")
            out += _okunus(int(tam)) + (["virgül"] + _okunus(int(kesir)) if kesir else [])
        else:
            out.append(p)
    esle = str.maketrans("âîûı", "aiui")
    return [w.translate(esle) for w in out]


def benzerlik(metin: str, duyulan_: str) -> float:
    return SequenceMatcher(None, _sade(metin), _sade(duyulan_)).ratio()


def duyulan(x: np.ndarray, sr: int) -> str:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        import soxr
        x = soxr.resample(x, sr, 16000).astype(np.float32)
    parcalar, _ = _whisper().transcribe(x, language="tr", beam_size=5)
    return " ".join(p.text.strip() for p in parcalar)


def dogruluk(x: np.ndarray, sr: int, metin: str) -> tuple[float, str]:
    d = duyulan(x, sr)
    return benzerlik(metin, d), d
