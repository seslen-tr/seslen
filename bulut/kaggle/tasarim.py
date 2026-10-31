# -*- coding: utf-8 -*-
"""
KAGGLE TASARIMI — kadronun (ogretmen/kadro.py) aday seslerini üretir.

Her aday için:
  00: tarif + DENEME[0] → kimlik kaydı (ref.wav). Kimlik temiz olmalı: Whisper
      kapısından (doğruluk ≥ 0,90, netlik ≥ 18) geçmezse başka tohumla ≤ 3 kez.
  01..: ref.wav referans + duygu tarifi (ogretmen/duygu.py) — üretimdeki ayarla
      (yalnız referans, adım 6): dinlenen ses, eğitim verisinin sesi olsun.
Çıktı: /kaggle/working/tasarim/<aday>/{ref.wav, NN.flac, olcum.json}
kaggleIs.py tasarim-indir → cikti/adaylar/<aday>/ ; kadro sayfası dinleti/kadro.html.
"""

import glob
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

GIRDI = Path(glob.glob("/kaggle/input/**/korpus.jsonl", recursive=True)[0]).parent
CIKTI = Path("/kaggle/working/tasarim")
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

istenen = ayar.get("adaylar") or [a for s in kadro.KADRO.values() if not s["secilen"] for a in s["adaylar"]]


def olc(wav, metin):
    d, duy = olcu.dogruluk(wav, sr, metin)
    return {"dogruluk": round(d, 3), "netlik": round(olcu.netlik(wav, sr), 1), "duyulan": duy,
            "sn": round(len(wav) / sr, 2)}


for aday in istenen:
    d = CIKTI / aday
    d.mkdir(parents=True, exist_ok=True)
    tarif = kadro.tarif(aday)
    kayit = []
    for i, (duygu_adi, cumle) in enumerate(kadro.DENEME):
        t0 = time.perf_counter()
        for deneme in range(3 if i == 0 else 1):
            torch.manual_seed(zlib.crc32(f"{aday}|{i}|{deneme}".encode()))
            if i == 0:
                wav = model.generate(text=f"({tarif}){cumle}", cfg_value=2.0, inference_timesteps=10)
            else:
                wav = model.generate(text=duygu.metne_ekle(duygu_adi, cumle), reference_wav_path=str(d / "ref.wav"),
                                     cfg_value=2.0, inference_timesteps=6)
            o = olc(wav, cumle)
            if o["dogruluk"] >= 0.90 and o["netlik"] >= 18:
                break
        if i == 0:
            sf.write(d / "ref.wav", wav, sr)
        sf.write(d / f"{i:02d}.flac", wav, sr, subtype="PCM_16")
        o.update(cumle=cumle, duygu=duygu_adi, dosya=f"{i:02d}.flac", deneme=deneme + 1,
                 uretim_sn=round(time.perf_counter() - t0, 1))
        kayit.append(o)
        print(f"{aday} {i} {duygu_adi:7s} doğ {o['dogruluk']:.2f} net {o['netlik']} "
              f"({o['uretim_sn']} sn, {deneme + 1} deneme) | {o['duyulan']}", flush=True)
    (d / "olcum.json").write_text(json.dumps({"tarif": tarif, "ref_metin": kadro.DENEME[0][1], "olcum": kayit},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
print(f"bitti: {len(istenen)} aday, {(time.time() - t_bas) / 60:.0f} dk", flush=True)
