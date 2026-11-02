# -*- coding: utf-8 -*-
"""
EZGİ BANKASI (02.11) — öğrenci sesi kendisi üretir, ezgiyi (hece süreleri + perde çizgisi) öğretmenin gerçek
kayıtlarından alır. Eğitim yok.

Öğrencinin sesi iyi, ezgisi düz: aynı cümlenin öğretmen ezgisi verilince hece perdesinin öğretmenle ilişkisi
0,62 → 0,90, süresininki 0,83 → 0,92; anlaşılırlık (CTC) ve UTMOS aynı kalıyor (araclar/deney/prozodi.py, 6 ses ×
5 doğrulama cümlesi). Başka bir cümlenin ezgisi ise (hece yapısı en benzer cümle) ölçüde öğrencinin kendi ezgisinden
kötü (perde r 0,39, takılan 11 → 14), kulakla "güzel ama öğretmen gibi değil" — bu yüzden isteğe bağlı.

  banka   öğretmen kayıtları (veri/egitim/wavs, bütün eğitim listelerinin vurgulu okunuşu) öğrencinin kendi
          hizalayıcısıyla (text_aligner + tekdüze en iyi yol, eğitimdeki gibi) işaretlerine bağlanır, perde
          çıkarıcıyla (JDC) F0 alınır. Hece başına süre ve 5 noktalı perde çizgisi (cümlenin ortanca perdesine göre
          yarım ton) cikti/ezgi/<ses>.npz + .json'a yazılır (pickle yok: indirilen banka kod çalıştıramaz).
  bul     "tam": aynı okunuş, aynı ton → o kaydın ezgisi. "benzer": yoksa aynı bitişli (. ? !) cümleler arasından
          hece yapısı (vurgu, sözcük ve öbek sonu, konum; turkce-vurgu'nun vurgu işaretinden) en yakın olan.
  uygula  hece süreleri (γ = 0,5 ile öğrencininkiyle karışık; konuşma hızı korunur) model.konus'un uzat'ına,
          perde çizgisi çözücüye giden F0'a (sesli çerçevelerde; perde ve tonlama ayarı üstüne uygulanır).

    python ogrenci/ezgi.py banka [ses …]       (varsayılan: bütün sesler; yodas ve doğrulama listesi alınmaz)
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
BANKA = KOK / "cikti" / "ezgi"
WAVS = KOK / "veri" / "egitim" / "wavs"
ORAN = 24000
NOKTA = 5                   # hece başına perde noktası
NOKTALAMA = set("?.,!;:…")
GAMMA = 0.5                 # hece süresinde öğretmenin payı
BETA = 1.0                  # perdede öğretmenin payı
AGIRLIK = np.array([1.0, 0.6, 1.0, 0.3, 0.5], np.float32)    # vurgu, sözcük sonu, öbek sonu, sözcük başı, konum


# ── hece ──────────────────────────────────────────────────────────────────────────────────────────────
def heceler(okunus: str, sozluk: dict) -> tuple[list[dict], dict[int, int]]:
    """Okunuşun işaret sırasıyla (baştaki dolgu 0, ilk karakter 1) heceleri. Her hece: isaret (sıra listesi),
    vurgu, sozcuk_basi, sozcuk_sonu, obek_sonu (sözcükten sonra noktalama). Ayrıca ara: sözcük sonrası boşluk ve
    noktalama işaretinin sırası → önceki hecenin sırası. Hece sınırı: vurgu işareti (turkce-vurgu hecenin başına
    koyar), yoksa iki ünlü arasındaki son ünsüzün önü."""
    from model import _UNLU
    ses, i = [], 1
    for c in okunus:
        if c in sozluk:
            ses.append((i, c))
            i += 1
    sonuc: list[dict] = []
    ara: dict[int, int] = {}
    sozcuk: list[tuple[int, str]] = []

    def bitir():
        if not sozcuk:
            return
        harf = [(k, c) for k, c in sozcuk if c not in NOKTALAMA]
        cekirdek = [j for j, (_, c) in enumerate(harf) if c in _UNLU]
        if cekirdek:
            sinir = [0]
            for a, b in zip(cekirdek, cekirdek[1:]):
                arada = range(a + 1, b)
                vurgu = [j for j in arada if harf[j][1] == "ˈ"]
                unsuz = [j for j in arada if harf[j][1] not in "ˈː"]
                sinir.append(vurgu[0] if vurgu else unsuz[-1] if unsuz else b)
            sinir.append(len(harf))
            for h, (a, b) in enumerate(zip(sinir, sinir[1:])):
                parca = harf[a:b]
                sonuc.append({"isaret": [k for k, _ in parca], "vurgu": any(c == "ˈ" for _, c in parca),
                              "sozcuk_basi": h == 0, "sozcuk_sonu": b == len(harf), "obek_sonu": False})
            if any(c in NOKTALAMA for _, c in sozcuk):
                sonuc[-1]["obek_sonu"] = True
            for k, c in sozcuk:
                if c in NOKTALAMA:
                    ara[k] = len(sonuc) - 1
        sozcuk.clear()

    for k, c in ses:
        if c == " ":
            bitir()
            if sonuc:
                ara[k] = len(sonuc) - 1
        else:
            sozcuk.append((k, c))
    bitir()
    if sonuc:
        sonuc[-1]["obek_sonu"] = True
    return sonuc, ara


def bitis(okunus: str) -> str:
    s = okunus.strip()
    return "?" if s.endswith("?") else "!" if s.endswith("!") else "."


def ozellik(hece: list[dict]) -> np.ndarray:
    n = len(hece)
    return np.array([[h["vurgu"], h["sozcuk_sonu"], h["obek_sonu"], h["sozcuk_basi"], i / max(1, n - 1)]
                     for i, h in enumerate(hece)], np.float32).reshape(-1, 5)


# ── öğretmen kaydından ezgi ───────────────────────────────────────────────────────────────────────────
def _yol(v: np.ndarray) -> np.ndarray:
    """Tekdüze en iyi yol (glow-tts maximum_path): v (N işaret, T çerçeve) → 0/1 hizalama, her işarete ≥1 çerçeve."""
    n, t = v.shape
    q = np.full((n, t), -np.inf)
    q[0, 0] = v[0, 0]
    for j in range(1, t):
        onceki = q[:, j - 1]
        q[:, j] = v[:, j] + np.maximum(onceki, np.concatenate([[-np.inf], onceki[:-1]]))
    yol = np.zeros((n, t), np.float32)
    i = n - 1
    for j in range(t - 1, -1, -1):
        yol[i, j] = 1
        if i > 0 and (j == i or q[i - 1, j - 1] > q[i, j - 1]):
            i -= 1
    return yol


def mel(ogr, x: np.ndarray):
    """Eğitimdekiyle aynı mel (model.Ogrenci.stil)."""
    import torch
    if not hasattr(ogr, "_mel"):
        import torchaudio
        ogr._mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    return (torch.log(1e-5 + ogr._mel(torch.from_numpy(x.astype(np.float32)))) + 4) / 4


def perde(ogr, x: np.ndarray) -> np.ndarray:
    """Sesin çerçeve F0'ı (12,5 ms, öğrencinin perde çıkarıcısı)."""
    import torch
    me = mel(ogr, x)
    with torch.no_grad():
        return ogr.model.pitch_extractor(me[:, : me.shape[1] // 2 * 2].unsqueeze(0).unsqueeze(1))[0].numpy().reshape(-1)


def cikar(ogr, x: np.ndarray, okunus: str) -> dict | None:
    """Öğretmen kaydı → işaret süreleri (birim 25 ms) ve çerçeve F0'ı (12,5 ms), öğrencinin hizalayıcısıyla."""
    import torch
    from utils import length_to_mask
    m = ogr.model
    t = torch.LongTensor([[0, *ogr.temiz(okunus), 0]])
    me = mel(ogr, x)
    me = me[:, : me.shape[1] // 2 * 2].unsqueeze(0)
    if me.shape[-1] // 2 < t.shape[1]:
        return None
    with torch.no_grad():
        _, _, s2s = m.text_aligner(me, length_to_mask(torch.LongTensor([me.shape[-1] // 2])), t)
        s2s = s2s.transpose(-1, -2)[..., 1:].transpose(-1, -2)[0].numpy()
        f0 = m.pitch_extractor(me.unsqueeze(1))[0].numpy().reshape(-1)
    return {"sure": _yol(s2s).sum(-1).astype(np.int32), "f0": f0.astype(np.float32)}


def hece_ezgisi(okunus: str, sure: np.ndarray, f0: np.ndarray, sozluk: dict) -> dict:
    """İşaret süreleri + çerçeve F0 → hece başına süre (birim) ve 5 noktalı perde çizgisi (ortancaya göre yarım ton)."""
    hece, ara = heceler(okunus, sozluk)
    bas = np.concatenate([[0], np.cumsum(sure)])
    sesli = f0[f0 > 40]
    orta = float(np.median(sesli)) if len(sesli) else 200.0
    sureler, cizgi = [], []
    for h in hece:
        a, b = bas[h["isaret"][0]], bas[h["isaret"][-1] + 1]
        sureler.append(float(b - a))
        p = f0[2 * a: 2 * b]
        v = np.where(p > 40)[0]
        cizgi.append(np.interp(np.linspace(0, len(p) - 1, NOKTA), v, 12 * np.log2(p[v] / orta)) if len(v) >= 2
                     else np.full(NOKTA, np.nan))
    aralar = np.zeros(len(hece), np.float32)
    for k, j in ara.items():
        aralar[j] += sure[k]
    return {"hece": hece, "sure": np.array(sureler, np.float32), "cizgi": np.array(cizgi, np.float32).reshape(-1, NOKTA),
            "ara": aralar}


def egitim_okunuslari(dogrulama_haric: bool = True) -> dict[str, str]:
    """Öğretmen kaydı → vurgulu okunuş, bütün eğitim listelerinden; aynı kayıtta en yeni listenin okunuşu. Gerçek
    kişi verisi (yodas) alınmaz; t10 doğrulama listesi (deneyin sınama cümleleri) de."""
    sira = ["egitim", "egitim_vurgu", "egitim_vurgu_on", "egitim_vurgu_serdar", "egitim_vurgu_hiz",
            "egitim_vurgu_odak", "egitim_vurgu_t6", "egitim_vurgu_t7", "egitim_vurgu_t8", "egitim_vurgu_t9",
            "egitim_vurgu_t10"]
    val_yol = KOK / "veri" / "egitim_vurgu_t10" / "val_list.txt"
    val = {s.split("|")[0] for s in val_yol.read_text(encoding="utf-8").splitlines() if s} \
        if dogrulama_haric and val_yol.exists() else set()
    sonuc: dict[str, str] = {}
    for ad in sira:
        for dosya in sorted((KOK / "veri" / ad).glob("*_list.txt")):
            for s in dosya.read_text(encoding="utf-8").splitlines():
                p = s.split("|")
                if len(p) >= 2 and "ˈ" in p[1] and not p[0].startswith("yodas/") and p[0] not in val:
                    sonuc[p[0]] = p[1]
    return sonuc


def banka_kur(sesler: list[str] | None = None, kok: Path = BANKA) -> None:
    import soundfile as sf
    from model import Ogrenci
    from studyo import URUN_AGIRLIK
    kok.mkdir(parents=True, exist_ok=True)
    ogr = Ogrenci(KOK / "cikti" / "ogrenci" / URUN_AGIRLIK, onnx=False)
    sozluk = ogr.temiz.word_index_dictionary
    tonlar = json.loads((KOK / "veri" / "egitim" / "ozet.json").read_text(encoding="utf-8"))["duygu"]
    kayitlar = egitim_okunuslari()
    sesler = sesler or sorted({d.split("/")[0] for d in kayitlar})
    for ses in sesler:
        if (kok / f"{ses}.npz").exists():
            continue
        meta, oz, sure, cizgi, ofset, t0 = [], [], [], [], [0], time.perf_counter()
        for dosya, okunus in ((d, o) for d, o in kayitlar.items() if d.startswith(ses + "/")):
            x, sr = sf.read(WAVS / dosya, dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != ORAN:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=ORAN)
            c = cikar(ogr, x, okunus)
            if c is None:
                continue
            e = hece_ezgisi(okunus, c["sure"], c["f0"], sozluk)
            if len(e["hece"]) < 2:
                continue
            meta.append({"dosya": dosya, "okunus": " ".join(okunus.split()),
                         "ton": tonlar.get(dosya.rsplit(".", 1)[0], "notr"), "bitis": bitis(okunus)})
            oz.append(ozellik(e["hece"]))
            sure.append(e["sure"])
            cizgi.append(e["cizgi"])
            ofset.append(ofset[-1] + len(e["hece"]))
        np.savez_compressed(kok / f"{ses}.npz", oz=np.concatenate(oz), sure=np.concatenate(sure),
                            cizgi=np.concatenate(cizgi), ofset=np.array(ofset, np.int64))
        (kok / f"{ses}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        print(f"{ses}: {len(meta)} cümle, {time.perf_counter() - t0:.0f} sn", flush=True)


# ── arama ─────────────────────────────────────────────────────────────────────────────────────────────
def dtw(a: np.ndarray, b: np.ndarray) -> tuple[float, list[list[int]]]:
    """a (n, 5) hedef, b (m, 5) aday → (ortalama maliyet, hedef hece başına eşleşen aday heceleri)."""
    c = (np.abs(a[:, None, :] - b[None, :, :]) * AGIRLIK).sum(-1)
    n, m = c.shape
    q = np.full((n + 1, m + 1), np.inf)
    q[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            q[i, j] = c[i - 1, j - 1] + min(q[i - 1, j - 1], q[i - 1, j], q[i, j - 1])
    i, j, eslesme, adim = n, m, [[] for _ in range(n)], 0
    while i > 0 and j > 0:
        eslesme[i - 1].append(j - 1)
        adim += 1
        k = int(np.argmin([q[i - 1, j - 1], q[i - 1, j], q[i, j - 1]]))
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return float(q[n, m] / adim), eslesme


class Banka:
    """Ses başına öğretmen ezgileri (cikti/ezgi/<ses>.npz + .json); ilk istenişte yüklenir. Banka yoksa hiçbir
    şey bulunmaz (ezgi öğrencinin kendisi)."""

    def __init__(self, kok: Path = BANKA):
        self.kok = kok
        self._ses: dict[str, dict | None] = {}
        self._kilit = threading.Lock()

    def _yukle(self, ses: str) -> dict | None:
        with self._kilit:
            if ses not in self._ses:
                yol = self.kok / f"{ses}.npz"
                if not yol.exists():
                    self._ses[ses] = None
                else:
                    v = np.load(yol, allow_pickle=False)
                    meta = json.loads((self.kok / f"{ses}.json").read_text(encoding="utf-8"))
                    dizin: dict[str, list[int]] = {}
                    for i, m in enumerate(meta):
                        dizin.setdefault(m["okunus"], []).append(i)
                    self._ses[ses] = {"meta": meta, "dizin": dizin, **{k: v[k] for k in v.files}}
            return self._ses[ses]

    def var_mi(self, ses: str) -> bool:
        return self._yukle(ses) is not None

    def _kayit(self, b: dict, i: int) -> dict:
        a, s = b["ofset"][i], b["ofset"][i + 1]
        return {"oz": b["oz"][a:s], "sure": b["sure"][a:s], "cizgi": b["cizgi"][a:s], **b["meta"][i]}

    def bul(self, ses: str, okunus: str, ton: str, kip: str, sozluk: dict,
            ayniyi_atla: bool = False) -> tuple[dict | None, dict]:
        """kip: "tam" (aynı okunuş, aynı ton) ya da "benzer" (yoksa en benzer cümle). → (ezgi, bilgi).
        ayniyi_atla: aynı okunuşlu kayıtlar aranmaz (deney: benzer cümlenin ezgisi ölçülürken)."""
        b = self._yukle(ses)
        if b is None or kip not in ("tam", "benzer"):
            return None, {}
        okunus = " ".join(okunus.split())
        ayni = [] if ayniyi_atla else [i for i in b["dizin"].get(okunus, []) if b["meta"][i]["ton"] == ton]
        if ayni:
            k = self._kayit(b, ayni[0])
            return {"sure": k["sure"] / max(1e-6, float(k["sure"].mean())), "cizgi": k["cizgi"]}, \
                {"ezgi": "ayni_cumle", "kaynak": k["dosya"]}
        if kip != "benzer":
            return None, {}
        hece, _ = heceler(okunus, sozluk)
        if len(hece) < 2:
            return None, {}
        oz, son, n = ozellik(hece), bitis(okunus), len(hece)
        uzunluk = np.diff(b["ofset"])
        aday = [i for i, m in enumerate(b["meta"]) if m["bitis"] == son and abs(int(uzunluk[i]) - n) <= max(2, n // 5)
                and not (ayniyi_atla and m["okunus"] == okunus)]
        ayni_ton = [i for i in aday if b["meta"][i]["ton"] == ton]
        aday = ayni_ton if len(ayni_ton) >= 20 else aday
        if not aday:
            return None, {}
        # ön eleme: adayın özelliklerini hedefin uzunluğuna örnekleyip L1 (ucuz), en iyi 20'ye DTW (pahalı)
        konum = np.linspace(0, 1, n)

        def kaba(i: int) -> float:
            o = self._kayit(b, i)["oz"]
            j = np.round(konum * (len(o) - 1)).astype(int)
            return float((np.abs(o[j] - oz) * AGIRLIK).sum() / n + 0.05 * abs(len(o) - n))
        en_iyi = None
        for i in sorted(aday, key=kaba)[:20]:
            k = self._kayit(b, i)
            puan, es = dtw(oz, k["oz"])
            puan += 0.05 * abs(len(k["oz"]) - n) + 0.2 * (k["ton"] != ton)
            if en_iyi is None or puan < en_iyi[0]:
                en_iyi = (puan, k, es)
        puan, k, es = en_iyi
        sure = np.array([np.mean(k["sure"][e]) for e in es], np.float32)
        with np.errstate(all="ignore"):
            cizgi = np.array([np.nanmean(k["cizgi"][e], axis=0) for e in es], np.float32)
        return {"sure": sure / max(1e-6, float(sure.mean())), "cizgi": cizgi}, \
            {"ezgi": "benzer_cumle", "kaynak": k["dosya"], "uzaklik": round(puan, 3)}


# ── uygulama (model.konus çağırır) ────────────────────────────────────────────────────────────────────
def sure_uzat(ogr, okunus: str, sure: np.ndarray, ezgi: dict, gamma: float = GAMMA) -> list[tuple[int, int, float]]:
    """Öğrencinin işaret süreleri → hece başına çarpan (model.hazirla'nın uzat biçiminde). Okunuşun ilk
    len(ezgi) hecesine uygulanır (kısa cümlede sona eklenen taşıyıcı cümleye dokunulmaz)."""
    hece = heceler(okunus, ogr.temiz.word_index_dictionary)[0][: len(ezgi["sure"])]
    if len(hece) < len(ezgi["sure"]) or gamma <= 0:
        return []
    u = np.array([sure[h["isaret"]].sum() for h in hece], np.float32)
    yeni = u ** (1 - gamma) * (u.mean() * ezgi["sure"]) ** gamma
    yeni *= u.sum() / max(1e-6, yeni.sum())                     # öğrencinin konuşma hızı korunur
    return [(h["isaret"][0], h["isaret"][-1] + 1, float(b / max(a, 1e-6))) for h, a, b in zip(hece, u, yeni)]


def perde_uygula(ogr, f0, hiza, okunus: str, ezgi: dict, beta: float = BETA):
    """Çözücüye giden F0 (1, 2F; torch) → hece hece öğretmenin perde çizgisine (cümlenin kendi ortancasına göre)."""
    import torch
    hece = heceler(okunus, ogr.temiz.word_index_dictionary)[0][: len(ezgi["cizgi"])]
    if len(hece) < len(ezgi["cizgi"]) or beta <= 0:
        return f0
    f = f0[0].numpy().astype(np.float64)
    sesli = f > 40
    if not sesli.any():
        return f0
    orta = float(np.median(f[sesli]))
    bas = np.concatenate([[0], np.cumsum(hiza[0].sum(-1).long().numpy())])
    for h, c in zip(hece, ezgi["cizgi"]):
        if np.all(np.isnan(c)):
            continue
        a, b = 2 * bas[h["isaret"][0]], 2 * bas[h["isaret"][-1] + 1]
        if b - a < 2:
            continue
        c = np.interp(np.arange(b - a), np.linspace(0, b - a - 1, NOKTA), np.nan_to_num(c, nan=float(np.nanmean(c))))
        p, v = f[a:b], sesli[a:b]
        p[v] = np.exp((1 - beta) * np.log(p[v]) + beta * np.log(orta * 2 ** (c[v] / 12)))
    yumusak = np.convolve(np.where(sesli, f, 0), np.ones(5) / 5, "same")      # hece sınırında basamak olmasın
    agirlik = np.convolve(sesli.astype(np.float64), np.ones(5) / 5, "same")
    f = np.where(sesli, yumusak / np.maximum(agirlik, 1e-6), f)
    return torch.from_numpy(f.astype(np.float32)).unsqueeze(0)


if __name__ == "__main__":
    sys.path[:0] = [str(KOK / "ogrenci")]
    if len(sys.argv) > 1 and sys.argv[1] == "banka":
        banka_kur(sys.argv[2:] or None)
    else:
        print(__doc__)
