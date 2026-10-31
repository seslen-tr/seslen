# -*- coding: utf-8 -*-
"""
SERDAR DUYGU DENEMESİ (08.10) — öğretmen kayıtlarında Serdar'ın neşeli / uyarı / merakı nötrden zor ayrılıyor
(etki büyüklüğü neşeli 0,74; Damla 2,54): VoxCPM2 "kalın, otoriter, sakin" kimlik kaydını tarife baskın tutmuş.
Yeniden üretmeden önce hangi tarif ve ayarın duyguyu gerçekten getirdiği burada, işlemcide, birkaç cümleyle denenir.

Her duygu × cümle için: nötr (tarifsiz) · mevcut tarif · güçlü tarif · güçlü tarif + cfg 3 (tarife daha sıkı uyar).
Ölçü aynı cümlenin nötrüne göre (eşli fark): perde, perde dalgalanması, güç, hız; Whisper doğruluğu (tarif sesli
okunursa düşer). Çıkış: cikti/serdar_dene/*.wav, cikti/serdar_dene.json, dinleti/serdar.html

    python ogretmen/serdarDene.py
"""
from __future__ import annotations

import html
import json
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

KOK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(KOK / "araclar" / "olcu"), str(KOK / "ogretmen")]
from duygu import TARIF  # noqa: E402

GUCLU = {
    "neseli": "very happy and joyful, bright smiling voice, lively and energetic, higher pitch, expressive melody",
    "uyari": "urgent and alarmed warning, loud, tense and commanding, emphatic, raised voice",
    "merak": "very curious and intrigued, wondering, strongly rising questioning intonation, expressive",
}
CUMLE = {
    "neseli": ["Bugün harika bir haber aldım, terfi ettim!", "Ne güzel bir sabah, herkes çok mutlu.",
               "Sonunda tatile çıkıyoruz, çok sevindim!"],
    "uyari": ["Dikkat edin, köprü her an çökebilir.", "Sakın o kabloya dokunma, elektrik var!",
              "Lütfen hemen binayı boşaltın."],
    "merak": ["Bu kutunun içinde ne var acaba?", "Dün akşam nereye gittin?", "Bu kapı nereye açılıyor, hiç merak ettin mi?"],
}
ADIM = 10
CIKTI = KOK / "cikti" / "serdar_dene"


def ozellik(x: np.ndarray, sr: int, metin: str) -> np.ndarray:
    import librosa
    x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=16000)
    x, _ = librosa.effects.trim(x, top_db=35)
    # kalın ses: pyin oktav sıçramasın diye dar aralık (Serdar nötr ~150 Hz)
    f0, sesli, _ = librosa.pyin(x, fmin=55, fmax=400, sr=16000, frame_length=1024, hop_length=256)
    f = f0[sesli & ~np.isnan(f0)]
    yt = 12 * np.log2(f / 100) if len(f) > 5 else np.array([np.nan])
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
    harf = sum(c.isalpha() for c in metin)
    return np.array([np.nanmedian(yt), np.nanstd(yt), 20 * np.log10(np.mean(rms) + 1e-6), harf / max(len(x) / 16000, .1)])


def main() -> int:
    import kadro
    from olcu import dogruluk
    from yukle import ogretmen
    model = ogretmen("cpu")
    sr = model.tts_model.sample_rate
    ref = str(KOK / "cikti" / "tazele" / kadro.secilenler()["serdar"] / "ref2.wav")
    CIKTI.mkdir(parents=True, exist_ok=True)
    turler = {"notr": lambda d: ("", 2.0), "mevcut": lambda d: (TARIF[d], 2.0),
              "guclu": lambda d: (GUCLU[d], 2.0), "guclu_cfg3": lambda d: (GUCLU[d], 3.0)}
    sonuc: dict = {}
    for duygu, cumleler in CUMLE.items():
        for i, metin in enumerate(cumleler):
            oz = {}
            for tur, f in turler.items():
                tarif, cfg = f(duygu)
                yol = CIKTI / f"{duygu}-{i}-{tur}.wav"
                if not yol.exists():
                    torch.manual_seed(zlib.crc32(f"serdar|{duygu}|{i}".encode()))
                    t0 = time.time()
                    w = model.generate(text=f"({tarif}){metin}" if tarif else metin, reference_wav_path=ref,
                                       cfg_value=cfg, inference_timesteps=ADIM)
                    sf.write(yol, w, sr)
                    print(f"  {yol.name} {time.time() - t0:.0f} sn", flush=True)
                w, _ = sf.read(yol, dtype="float32")
                d, duyulan = dogruluk(w, sr, metin)
                oz[tur] = (ozellik(w, sr, metin), d, duyulan)
            for tur in list(turler)[1:]:
                fark = oz[tur][0] - oz["notr"][0]
                sonuc.setdefault(duygu, {}).setdefault(tur, []).append(
                    {"cumle": metin, "perde": round(float(fark[0]), 2), "dalga": round(float(fark[1]), 2),
                     "guc": round(float(fark[2]), 2), "hiz": round(float(fark[3]), 2), "dogruluk": round(oz[tur][1], 2),
                     "duyulan": oz[tur][2]})
            print(duygu, i, "tamam", flush=True)
    ozet = {d: {t: {k: round(float(np.nanmean([x[k] for x in v])), 2) for k in ("perde", "dalga", "guc", "hiz", "dogruluk")}
                for t, v in tv.items()} for d, tv in sonuc.items()}
    (KOK / "cikti" / "serdar_dene.json").write_text(json.dumps({"ozet": ozet, "ayrinti": sonuc}, ensure_ascii=False,
                                                               indent=1), encoding="utf-8")
    for d, tv in ozet.items():
        for t, v in tv.items():
            print(f"{d:7s} {t:11s} perde {v['perde']:+.2f} yt  dalga {v['dalga']:+.2f}  güç {v['guc']:+.2f} dB  "
                  f"hız {v['hiz']:+.2f} harf/sn  doğruluk {v['dogruluk']:.2f}", flush=True)

    # dinleme sayfası
    hedef = KOK / "dinleti" / "ses"
    hedef.mkdir(parents=True, exist_ok=True)
    e = html.escape
    ad = {"notr": "Nötr", "mevcut": "Bugünkü tarif", "guclu": "Güçlü tarif", "guclu_cfg3": "Güçlü + sıkı"}
    satir = []
    for duygu, cumleler in CUMLE.items():
        for i, metin in enumerate(cumleler):
            hucre = ""
            for tur in turler:
                shutil.copy(CIKTI / f"{duygu}-{i}-{tur}.wav", hedef / f"serdar-{duygu}-{i}-{tur}.wav")
                hucre += f"<td><audio controls preload=none src='ses/serdar-{duygu}-{i}-{tur}.wav'></audio></td>"
            satir.append(f"<tr><th>{duygu}</th><td>{e(metin)}</td>{hucre}</tr>")
    (KOK / "dinleti" / "serdar.html").write_text(f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Serdar duygu denemesi</title><style>
:root {{ --zemin:#f4f5f8; --yuzey:#fff; --metin:#1b2230; --soluk:#5f6b7c; --cizgi:#dde2ea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --zemin:#0f1216; --yuzey:#171c23; --metin:#e8edf2; --soluk:#8b98a8; --cizgi:#2a323d; }} }}
body {{ margin:0; background:var(--zemin); color:var(--metin); font:14px/1.45 system-ui, sans-serif; }}
main {{ max-width:1300px; margin:0 auto; padding:24px 16px 64px; }} p {{ color:var(--soluk); }}
.kaydir {{ overflow-x:auto; }} table {{ border-collapse:collapse; background:var(--yuzey); }}
th, td {{ border-bottom:1px solid var(--cizgi); padding:6px 8px; text-align:left; }} audio {{ width:200px; height:30px; }}
</style></head><body><main><h1>Serdar — öğretmen duygu denemesi</h1>
<p>Her satır aynı cümle: nötr, bugünkü tarif (eğitim verisi böyle üretildi), güçlü tarif, güçlü tarif + sıkı uyum.
Hangi sütunda duygu belirgin ve ses hâlâ Serdar?</p>
<div class=kaydir><table><tr><th>duygu</th><th>cümle</th>{''.join(f'<th>{v}</th>' for v in ad.values())}</tr>
{''.join(satir)}</table></div></main></body></html>""", encoding="utf-8")
    print("dinleti/serdar.html yazıldı")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
