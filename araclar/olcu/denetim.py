# -*- coding: utf-8 -*-
"""
OTOMATİK DENETİM (16.10) — öğretmen verisini kulakla tek tek dinlemek mümkün değil. Whisper bozuk sözcüğü de doğru
yazıyor (dil modeliyle tamamlıyor; kelimeGuven.py, 01.10), bu yüzden dil modelsiz, harf harf yazan bir Türkçe CTC
tanıyıcı (wav2vec2 XLS-R 300M, Common Voice 8 Türkçe) kullanılır; bozuk sözcük orada bozuk görünür.

Kayıt başına:
  gop       sözcüğün en kötü harfinin telaffuz puanı (GOP): metin CTC ile zorunlu hizalanır, her harfin karelerinde
            log P(metindeki harf) − log P(en olası harf) ortalaması. 0 kusursuz; −5'in altı bozuk okunuş
            ("harika" → "heriga" −6,3, "okula" → "o kıla" −5,5). Metin yazım hatasını da bulur ("tamamlanldi").
  sozcuk    her sözcüğün harf hata oranı (metinle harf düzeyinde hizalanır; birleşik/ayrık yazım hata sayılmaz)
  fazla     metnin başında / sonunda tanıyıcının duyduğu fazladan harf ("ha", "am", "aha": kahkaha, mırıltı)
  eksik     okunmayan sözcük
Bayrak: GOP < ESIK_GOP, başta/sonda fazladan ≥ 2 harf, ya da okunmayan sözcük. Duygu ve ton tutarlılığı ayrı.

    python araclar/olcu/denetim.py cikti/kaggle/indirilen/seslen-yeniden-uretim-1/veri/ayse [...]
Çıktı: <klasör>/denetim.jsonl ({"kimlik", "duyulan", "en_kotu", "fazla", "eksik", "bayrak", "neden"}); kaldığı yerden sürer.
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL = "mpoyraz/wav2vec2-xls-r-300m-cv8-turkish"
ESIK_GOP = -5.0          # "okula" → "o kıla" −5,48
_HARF = re.compile(r"[^a-zçğıöşüâîû ]")


def normal(t: str) -> str:
    t = t.replace("I", "ı").replace("İ", "i").lower().replace("â", "a").replace("î", "i").replace("û", "u")
    return " ".join(_HARF.sub(" ", t.replace("'", "").replace("’", "")).split())


CIHAZ = "cpu"                      # Kaggle'da yeniden.py "cuda:1" yapar


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    torch.set_num_threads(4)
    return Wav2Vec2Processor.from_pretrained(MODEL), Wav2Vec2ForCTC.from_pretrained(MODEL).eval().to(CIHAZ)


def _logp(x: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    import torch
    p, m = _model()
    x = x.mean(axis=1) if x.ndim > 1 else x
    x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=16000)
    with torch.no_grad():
        g = p(x, sampling_rate=16000, return_tensors="pt").input_values.to(CIHAZ)
        return torch.log_softmax(m(g).logits[0].float(), -1).cpu().numpy()


def duy(x: np.ndarray, sr: int, lp: np.ndarray | None = None) -> str:
    p, _ = _model()
    lp = _logp(x, sr) if lp is None else lp
    return normal(p.batch_decode(lp.argmax(-1)[None])[0])


def gop(lp: np.ndarray, metin: str) -> list[tuple[str, float]]:
    """Sözcük başına en kötü harfin puanı (CTC Viterbi zorunlu hizalaması)."""
    p, _ = _model()
    voc, blank = p.tokenizer.get_vocab(), p.tokenizer.pad_token_id
    ref = normal(metin)
    hedef = [voc.get("|") if c == " " else voc.get(c, voc.get("|")) for c in ref]
    S = [blank]
    for t in hedef:
        S += [t, blank]
    T, L = lp.shape[0], len(S)
    if T < L // 2 + 1:
        return [(w, -20.0) for w in ref.split()]
    e = lp[:, S]
    d = np.full((T, L), -1e9)
    bp = np.zeros((T, L), dtype=np.int32)
    d[0, :2] = e[0, :2]
    atla = np.array([s >= 2 and S[s] != blank and S[s] != S[s - 2] for s in range(L)])
    for t in range(1, T):
        a = d[t - 1]
        b = np.concatenate([[-1e9], a[:-1]])
        c = np.where(atla, np.concatenate([[-1e9, -1e9], a[:-2]]), -1e9)
        yig = np.stack([a, b, c])
        k = yig.argmax(0)
        d[t] = yig[k, np.arange(L)] + e[t]
        bp[t] = np.arange(L) - k
    s = L - 1 if d[T - 1, L - 1] > d[T - 1, L - 2] else L - 2
    en_iyi = lp.max(1)
    harf: dict[int, list[float]] = {}
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:
            harf.setdefault(s // 2, []).append(lp[t, S[s]] - en_iyi[t])
        s = bp[t, s]
    puan = [float(np.mean(harf[i])) if i in harf else -10.0 for i in range(len(hedef))]
    sonuc, i = [], 0
    for w in ref.split():
        sonuc.append((w, round(min(puan[i:i + len(w)]), 2)))
        i += len(w) + 1
    return sonuc


def hizala(ref: str, hyp: str) -> list[tuple[str | None, str | None]]:
    """Harf düzeyinde Levenshtein hizası (boşluklar yok sayılır): [(ref harfi, hyp harfi)], None = boşluk."""
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    yol, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            yol.append((r[i - 1], h[j - 1])); i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            yol.append((r[i - 1], None)); i -= 1
        else:
            yol.append((None, h[j - 1])); j -= 1
    return yol[::-1]


def puanla(metin: str, duyulan: str, g: list[tuple[str, float]] | None = None) -> dict:
    ref = normal(metin)
    yol = hizala(ref, duyulan)
    sozcukler = ref.split()
    sinir, k = [], 0                                   # her sözcüğün ref harf aralığı
    for s in sozcukler:
        sinir.append((k, k + len(s)))
        k += len(s)
    ri, hata = 0, [0] * len(sozcukler)
    bas = son = 0
    for a, b in yol:
        if a is None:                                  # fazladan harf
            if ri == 0:
                bas += 1
            elif ri == k:
                son += 1
            else:
                w = next(i for i, (x, y) in enumerate(sinir) if x <= ri - 1 < y)
                hata[w] += 1
            continue
        w = next(i for i, (x, y) in enumerate(sinir) if x <= ri < y)
        hata[w] += (b is None) or (a != b)
        ri += 1
    oran = [h / max(1, y - x) for h, (x, y) in zip(hata, sinir)]      # harf hata oranı: yalnız bilgi (GOP karar verir)
    eksik = [s for s, o in zip(sozcukler, oran) if o >= 1.0 and len(s) >= 3]     # "bu", "mı": CTC yutabiliyor
    neden = []
    if bas >= 2:
        neden.append(f"başta fazladan {bas} harf")
    if son >= 2:
        neden.append(f"sonda fazladan {son} harf")
    en_kotu = max((o for s, o in zip(sozcukler, oran) if len(s) >= 3), default=0.0)
    g = g or []
    neden += [f"{w}:{v}" for w, v in g if v < ESIK_GOP]
    neden += [f"okunmadı:{w}" for w in eksik]
    return {"duyulan": duyulan, "en_kotu": round(en_kotu, 3), "gop": min((v for _, v in g), default=0.0),
            "fazla": [bas, son], "eksik": eksik, "bayrak": bool(neden), "neden": neden}


def denetle(x: np.ndarray, sr: int, metin: str) -> dict:
    lp = _logp(x, sr)
    return puanla(metin, duy(x, sr, lp), gop(lp, metin))


def klasor(d: Path) -> None:
    import soundfile as sf
    cikti = d / "denetim.jsonl"
    bitti = {json.loads(x)["kimlik"] for x in cikti.read_text(encoding="utf-8").splitlines() if x} if cikti.exists() else set()
    olcum = [json.loads(x) for x in (d / "olcum.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    olcum = [o for o in olcum if o["kimlik"] not in bitti and (d / "wavs" / f"{o['kimlik']}.flac").exists()]
    with open(cikti, "a", encoding="utf-8") as f:
        for i, o in enumerate(olcum, 1):
            x, sr = sf.read(d / "wavs" / f"{o['kimlik']}.flac", dtype="float32")
            lp = _logp(x, sr)
            f.write(json.dumps({"kimlik": o["kimlik"], **puanla(o["metin"], duy(x, sr, lp), gop(lp, o["metin"]))},
                               ensure_ascii=False) + "\n")
            if i % 200 == 0:
                f.flush()
                print(f"  {d.name}: {i}/{len(olcum)}", flush=True)
    sonuc = [json.loads(x) for x in cikti.read_text(encoding="utf-8").splitlines() if x]
    print(f"{d.name}: {len(sonuc)} kayıt, bayraklı %{100 * sum(s['bayrak'] for s in sonuc) / max(1, len(sonuc)):.1f}",
          flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        klasor(Path(a))
