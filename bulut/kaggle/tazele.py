# -*- coding: utf-8 -*-
"""
KAGGLE: KİMLİK TAZELEME + KARŞILAŞTIRMA — "daha kaliteli, Türkçeye yatkın" sorusu.

Seçilen her ses için (ogretmen/kadro.py):
  1) Tazeleme: kısa ref.wav (tasarımdan, ~4 sn) referans verilip TAZELE_METIN
     (~10 sn doğal Türkçe) 3 tohumla okunur; kapıdan geçen en net olan
     ref2.wav olur. Uzun, doğal Türkçe referans → tını kararlı, tonlama Türkçe.
  2) Karşılaştırma (KARSILASTIRMA cümleleri), üç kip:
       A  ref.wav, yalnız referans, adım 6   (şimdiki üretim ayarı)
       B  ref2.wav, yalnız referans, adım 10
       C  ref2.wav + devam kipi (prompt = ref2 + metni), adım 10
AYAR: {"sesler": {"ayse": "ayse-3", …}, "tasarla": ["serdar-4", …]}
  tasarla: önce tariften tasarlanır (tasarim.py gibi), sonra aynı iki adım.
Girdi: ref/aday/<aday>.wav (kaggleIs.py paketle). Çıktı: /kaggle/working/tazele/<aday>/
"""

import glob
import json
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

GIRDI = Path(glob.glob("/kaggle/input/**/korpus.jsonl", recursive=True)[0]).parent
CIKTI = Path("/kaggle/working/tazele")
t_bas = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "voxcpm==2.0.3", "faster-whisper==1.2.1",
                "soxr", "librosa"], check=True)
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

ayar = globals().get("AYAR") or {}
gpu_sayisi = torch.cuda.device_count()
sys.path.insert(0, str(GIRDI))
import duygu  # noqa: E402
import kadro  # noqa: E402
import olcu  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

olcu._whisper.cache_clear()
_w = WhisperModel("medium", device="cuda", device_index=1 if gpu_sayisi > 1 else 0, compute_type="int8_float16")
olcu._whisper = lambda: _w

from huggingface_hub import snapshot_download  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

if torch.cuda.get_device_capability(0)[0] < 8:          # T4/P100: bf16 yok → float32 (uret.py ile aynı)
    import voxcpm.model.voxcpm2 as _v2
    _v2.pick_runtime_dtype = lambda device, dtype: "float32"
model = VoxCPM.from_pretrained(snapshot_download("openbmb/VoxCPM2", local_dir="/kaggle/tmp/VoxCPM2"),
                               load_denoiser=False, optimize=False, device="cuda:0", local_files_only=True)
sr = model.tts_model.sample_rate
print(f"model hazır ({time.time() - t_bas:.0f} sn)", flush=True)


def olc(wav, metin):
    d, duy = olcu.dogruluk(wav, sr, metin)
    return {"dogruluk": round(d, 3), "netlik": round(olcu.netlik(wav, sr), 1), "duyulan": duy,
            "sn": round(len(wav) / sr, 2)}


def gecer(o):
    return o["dogruluk"] >= 0.90 and o["netlik"] >= 18


def uret(tohum, **k):
    torch.manual_seed(zlib.crc32(tohum.encode()))
    return model.generate(cfg_value=2.0, **k)


def tasarla(aday, d):
    metin = kadro.DENEME[0][1]
    for deneme in range(3):
        wav = uret(f"{aday}|0|{deneme}", text=f"({kadro.tarif(aday)}){metin}", inference_timesteps=10)
        o = olc(wav, metin)
        if gecer(o):
            break
    sf.write(d / "ref.wav", wav, sr)
    return o


isler = [(ses, aday, False) for ses, aday in ayar.get("sesler", {}).items()]
isler += [(aday.rsplit("-", 1)[0], aday, True) for aday in ayar.get("tasarla", [])]
for ses, aday, yeni in isler:
    d = CIKTI / aday
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    kayit = {"ses": ses, "aday": aday, "tarif": kadro.tarif(aday), "yeni": yeni}
    if yeni:
        kayit["tasarim"] = tasarla(aday, d)
    else:
        shutil.copy(next(GIRDI.glob(f"**/aday/{aday}.wav")), d / "ref.wav")
    ref = str(d / "ref.wav")

    adaylar = []
    for i in range(3):
        wav = uret(f"{aday}|tazele|{i}", text=kadro.TAZELE_METIN, reference_wav_path=ref, inference_timesteps=10)
        adaylar.append((olc(wav, kadro.TAZELE_METIN), wav))
    en = max(adaylar, key=lambda a: (gecer(a[0]), a[0]["netlik"]))
    sf.write(d / "ref2.wav", en[1], sr)
    kayit["tazele"] = en[0]
    kayit["tazele_adaylar"] = [a[0]["netlik"] for a in adaylar]
    ref2 = str(d / "ref2.wav")

    kip = {"A": dict(reference_wav_path=ref, inference_timesteps=6),
           "B": dict(reference_wav_path=ref2, inference_timesteps=10),
           "C": dict(reference_wav_path=ref2, prompt_wav_path=ref2, prompt_text=kadro.TAZELE_METIN,
                     inference_timesteps=10)}
    kayit["karsilastirma"] = []
    for n, (duygu_adi, cumle) in enumerate(kadro.KARSILASTIRMA):
        for k, arg in kip.items():
            wav = uret(f"{aday}|{n}|{k}", text=duygu.metne_ekle(duygu_adi, cumle), **arg)
            sf.write(d / f"{k}{n}.flac", wav, sr, subtype="PCM_16")
            o = olc(wav, cumle)
            o.update(kip=k, n=n, cumle=cumle, duygu=duygu_adi, dosya=f"{k}{n}.flac")
            kayit["karsilastirma"].append(o)
            print(f"{aday} {k}{n} doğ {o['dogruluk']:.2f} net {o['netlik']} | {o['duyulan']}", flush=True)
    (d / "olcum.json").write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{aday}: tazele net {en[0]['netlik']} doğ {en[0]['dogruluk']:.2f} "
          f"({time.perf_counter() - t0:.0f} sn)", flush=True)
print(f"bitti: {len(isler)} ses, {(time.time() - t_bas) / 60:.0f} dk", flush=True)
