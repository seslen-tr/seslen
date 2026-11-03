# -*- coding: utf-8 -*-
"""
SES DOKUSU TEŞHİSİ (02.11) — kullanıcının ezgi bankası yorumu: "güzel ama öğretmen gibi değil". Fark ezgide değilse
nerede? prozodi.py'nin 30 cümlesinde (t10 doğrulama, 6 ses × 5) beş okuyuş:
  ogrenci      öğrencinin her şeyi kendi tahmini
  ust_sinir    öğrenci + aynı cümlenin öğretmen ezgisi (stüdyonun "tam" kipi)
  hiza         çözücüye öğretmen kaydının kendi hizası; F0 ve N öğrencinin tahmini (o hizayla)
  hiza_maske   öğretmenin hizası; F0 tahmin ama öğretmenin sessiz çerçevelerinde 0 (sesli/sessiz yerleri mi?)
  hiza_puruzsuz öğretmenin hizası ve ~140 ms pürüzsüzleştirilmiş F0'ı (eğri mi, ince oynama mı?)
  hiza_f0      öğretmenin hizası ve F0'ı; N tahmin
  kurma_urun   çözücüye öğretmen kaydının kendi hizası, perdesi (F0) ve enerjisi (N); stil ürün stili
  kurma_oz     aynısı, stil o kaydın kendisinden (çözücünün tavanı: girdiler kusursuzken ne kadar öğretmen gibi?)
  ogretmen     öğretmenin kaydı
Ölçü (öğretmene göre): konuşmacı benzerliği (WavLM-SV kosinüs; başvuru: aynı sesin başka bir öğretmen kaydı),
mel-kepstrum uzaklığı (MCD, DTW ile; eğitim log-mel'inden, göreli — dB ile karşılaştırılmaz), UTMOS. kurma_oz da öğretmenden uzaksa cilacı çözücünün çıkışında çalışmalı;
yakınsa fark tahmin edilen özelliklerde (enerji, stil). Çıktı: cikti/doku/ (sonuc.json, ozet.json, dinle.html).

    python araclar/deney/doku.py [--adet 5]
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "araclar" / "olcu"), str(KOK / "araclar" / "deney")]
CIKTI = KOK / "cikti" / "doku"
ORAN = 24000
KIPLER = ("ogrenci", "ust_sinir", "hiza", "hiza_maske", "hiza_puruzsuz", "hiza_f0", "kurma_urun", "kurma_oz", "ogretmen")
SV = "microsoft/wavlm-base-plus-sv"


class Benzerlik:
    """Konuşmacı gömmesi (WavLM-SV, 16 kHz) — yalnız ölçü için."""

    def __init__(self):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.oz = AutoFeatureExtractor.from_pretrained(SV)
        self.model = WavLMForXVector.from_pretrained(SV).eval()

    def gomme(self, x: np.ndarray) -> np.ndarray:
        import librosa
        import torch
        g = self.oz(librosa.resample(x.astype(np.float32), orig_sr=ORAN, target_sr=16000), sampling_rate=16000,
                    return_tensors="pt")
        with torch.no_grad():
            e = self.model(**g).embeddings[0].numpy()
        return e / np.linalg.norm(e)


def kepstrum(ogr, x: np.ndarray) -> np.ndarray:
    """Eğitimdeki log-mel'den mel-kepstrum (c1..c24), çerçeve × 24."""
    from scipy.fft import dct
    from ezgi import mel
    lm = mel(ogr, x).numpy() * 4 - 4                     # (80, çerçeve), doğal log
    return dct(lm, type=2, norm="ortho", axis=0)[1:25].T


def mcd(a: np.ndarray, b: np.ndarray) -> float:
    import librosa
    from scipy.spatial.distance import cdist
    maliyet = cdist(a, b)
    _, yol = librosa.sequence.dtw(C=maliyet)
    return float(10 / np.log(10) * np.sqrt(2) * maliyet[yol[:, 0], yol[:, 1]].mean())


def puruzsuz(f0: np.ndarray, pencere: int = 11) -> np.ndarray:
    """Sesli bölümlerde F0'ın kayan ortalaması (~140 ms): eğri kalır, ince oynama gider; sessizler 0 kalır."""
    sesli = f0 > 40
    v = np.where(sesli, np.log(np.maximum(f0, 1)), 0)
    c = np.ones(pencere)
    pay, payda = np.convolve(v, c, "same"), np.convolve(sesli.astype(float), c, "same")
    return np.where(sesli, np.exp(pay / np.maximum(payda, 1)), 0).astype(np.float32)


def kur(ogr, okunus: str, x: np.ndarray, sure: np.ndarray, stil: np.ndarray, ref: np.ndarray | None = None,
        f0_ogretmen: bool | str = True, n_ogretmen: bool = True) -> np.ndarray:
    """Çözücüye öğretmen kaydının hizası; F0 ve enerji (N) öğretmenden ya da öğrencinin tahmini (bu hizayla).
    stil: ürün stili (256; tahmin ve çözücü stili), ref: çözücü stili yerine (128).
    f0_ogretmen "maske": tahmin F0, öğretmenin sessiz çerçevelerinde 0; "puruzsuz": öğretmen F0, ince oynaması alınmış."""
    import torch
    from ezgi import mel
    from utils import log_norm
    m = ogr.model
    d, t_en, _, s, ref0, _ = ogr.hazirla("", stil, okunus=okunus)
    hiza = torch.repeat_interleave(torch.eye(len(sure)), torch.from_numpy(sure.astype(np.int64)), dim=1).unsqueeze(0)
    me = mel(ogr, x)
    me = me[:, : int(sure.sum()) * 2].unsqueeze(0)
    with torch.no_grad():
        f0, nn = m.predictor.F0Ntrain(d.transpose(-1, -2) @ hiza, s)
        if f0_ogretmen:
            f0o = m.pitch_extractor(me.unsqueeze(1))[0].reshape(1, -1)
            if f0_ogretmen == "maske":
                f0 = f0 * (f0o > 40)
            elif f0_ogretmen == "puruzsuz":
                f0 = torch.from_numpy(puruzsuz(f0o[0].numpy())).unsqueeze(0)
            else:
                f0 = f0o
        if n_ogretmen:
            nn = log_norm(me.unsqueeze(1)).squeeze(1)
        w = m.decoder(t_en @ hiza, f0, nn, ref0 if ref is None else torch.from_numpy(ref).float().unsqueeze(0))
    return w.squeeze().numpy()[..., :-50]


def main(adet: int = 5) -> None:
    import soundfile as sf
    from dogallik import puan
    from ezgi import WAVS, cikar, hece_ezgisi, heceler
    from model import Ogrenci
    from prozodi import LISTE, SESLER
    from studyo import URUN_AGIRLIK
    ogr = Ogrenci(KOK / "cikti" / "ogrenci" / URUN_AGIRLIK)
    sozluk = ogr.temiz.word_index_dictionary
    tonlar = json.loads((KOK / "veri" / "egitim" / "ozet.json").read_text(encoding="utf-8"))["duygu"]
    sv = Benzerlik()
    val = [s.split("|") for s in (LISTE / "val_list.txt").read_text(encoding="utf-8").splitlines() if s]
    sonuc = []
    for ses in SESLER:
        (CIKTI / ses).mkdir(parents=True, exist_ok=True)
        secilen = [s for s in val if s[0].startswith(ses + "/") and len(heceler(s[1], sozluk)[0]) >= 4]
        for i, (dosya, okunus, _) in enumerate(secilen[:adet]):
            ad, ton = Path(dosya).stem, tonlar.get(dosya.rsplit(".", 1)[0], "notr")
            urun = ogr.cumle_stili(okunus, ogr.stil_dosyadan(ses, ton if f"{ses}/{ton}" in ogr.stiller else "notr"),
                                   okunus=okunus)
            x, _ = sf.read(WAVS / dosya, dtype="float32")
            c = cikar(ogr, x, okunus)
            if c is None:
                continue
            baska, _ = sf.read(WAVS / secilen[(i + 1) % len(secilen)][0], dtype="float32")
            e = hece_ezgisi(okunus, c["sure"], c["f0"], sozluk)
            okuyus = {"ogrenci": ogr.konus("", urun, okunus=okunus),
                      "ust_sinir": ogr.konus("", urun, okunus=okunus,
                                             ezgi={"sure": e["sure"] / e["sure"].mean(), "cizgi": e["cizgi"]}),
                      "hiza": kur(ogr, okunus, x, c["sure"], urun, f0_ogretmen=False, n_ogretmen=False),
                      "hiza_maske": kur(ogr, okunus, x, c["sure"], urun, f0_ogretmen="maske", n_ogretmen=False),
                      "hiza_puruzsuz": kur(ogr, okunus, x, c["sure"], urun, f0_ogretmen="puruzsuz", n_ogretmen=False),
                      "hiza_f0": kur(ogr, okunus, x, c["sure"], urun, n_ogretmen=False),
                      "kurma_urun": kur(ogr, okunus, x, c["sure"], urun),
                      "kurma_oz": kur(ogr, okunus, x, c["sure"], urun, ogr.stil(x)[:128]),
                      "ogretmen": x}
            g_ogr, k_ogr = sv.gomme(x), kepstrum(ogr, x)
            kayit = {"ses": ses, "dosya": dosya, "ton": ton, "okunus": okunus,
                     "baska_benzerlik": round(float(sv.gomme(baska) @ g_ogr), 3)}
            for kip, w in okuyus.items():
                yol = CIKTI / ses / f"{ad}_{kip}.wav"
                sf.write(yol, w, ORAN)
                kayit[kip] = {"benzerlik": round(float(sv.gomme(w) @ g_ogr), 3), "utmos": round(puan(yol), 2)}
                if kip != "ogretmen":
                    kayit[kip]["mcd"] = round(mcd(kepstrum(ogr, w), k_ogr), 2)
            sonuc.append(kayit)
            print(json.dumps({k: kayit[k] for k in ("ses", "baska_benzerlik", *KIPLER[:-1])}, ensure_ascii=False),
                  flush=True)
    (CIKTI / "sonuc.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    ozet = {"baska_ogretmen_kaydi": {"benzerlik": round(float(np.mean([k["baska_benzerlik"] for k in sonuc])), 3)}}
    for kip in KIPLER:
        ozet[kip] = {a: round(float(np.mean([k[kip][a] for k in sonuc])), 3) for a in sonuc[0][kip]}
    print(json.dumps(ozet, ensure_ascii=False, indent=1))
    (CIKTI / "ozet.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=1), encoding="utf-8")
    sayfa(sonuc, ozet)


def sayfa(sonuc: list[dict], ozet: dict) -> None:
    from prozodi import ipa_metin
    ad = {"ogrenci": "Öğrenci", "ust_sinir": "Öğrenci + öğretmen ezgisi", "hiza": "Öğretmenin hizası (F0, N tahmin)",
          "hiza_maske": "Öğretmenin hizası + sesli/sessiz yerleri", "hiza_puruzsuz": "Öğretmenin hizası + pürüzsüz F0",
          "hiza_f0": "Öğretmenin hizası + F0 (N tahmin)", "kurma_urun": "Çözücü, öğretmenin girdileri",
          "kurma_oz": "Çözücü, öğretmenin girdileri + kendi stili", "ogretmen": "Öğretmen"}
    tablo = "".join(f"<tr><td>{ad.get(k, 'Aynı sesin başka bir öğretmen kaydı')}</td><td>{v['benzerlik']}</td>"
                    f"<td>{v.get('mcd', '—')}</td><td>{v.get('utmos', '—')}</td></tr>" for k, v in ozet.items())
    satir = []
    for k in sonuc:
        a = Path(k["dosya"]).stem
        hucre = "".join(f"<td><audio controls preload=none src='{k['ses']}/{a}_{kip}.wav'></audio><br><small>"
                        f"benzerlik {k[kip]['benzerlik']}" + (f" · MCD {k[kip]['mcd']}" if "mcd" in k[kip] else "")
                        + f" · UTMOS {k[kip]['utmos']}</small></td>" for kip in KIPLER)
        satir.append(f"<tr><td><b>{k['ses']}</b> · {k['ton']}<br>{html.escape(ipa_metin(k['okunus']))}</td>{hucre}</tr>")
    (CIKTI / "dinle.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Ses dokusu</title><style>body{font:14px sans-serif;margin:16px}"
        "td,th{padding:4px 8px;border-bottom:1px solid #ddd;vertical-align:top;text-align:left}audio{height:30px;"
        "width:190px}</style><h1>Ses dokusu: öğretmenden fark nerede?</h1><p>benzerlik: öğretmen kaydıyla konuşmacı "
        "benzerliği (1 = aynı). MCD: mel-kepstrum uzaklığı (göreli, düşük = yakın).</p><table><tr><th>okuyuş</th>"
        "<th>benzerlik</th><th>MCD</th><th>UTMOS</th></tr>" + tablo + "</table><h2>Dinle</h2><table><tr><th>cümle</th>"
        + "".join(f"<th>{v}</th>" for v in ad.values()) + "</tr>" + "".join(satir) + "</table>", encoding="utf-8")
    print(CIKTI / "dinle.html")


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--adet") + 1]) if "--adet" in sys.argv else 5)
