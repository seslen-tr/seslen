# -*- coding: utf-8 -*-
"""
KAGGLE: ÖĞRETMEN DENEMESİ (13.10) — Hasan tariften tasarlanınca İngilizce aksanlı çıkıyor (hasan-3, hasan-6). Aynı
cümleleri (kadro.DENEME) farklı yollarla üretip kulakla karşılaştırmak için:
  chatterbox  Chatterbox Multilingual (MIT, Türkçe destekli): dil "tr", referansın ses rengi; cfg_weight 0 aksan
              aktarımını keser
  voxcpm      VoxCPM2: Türkçesi iyi bir referans + yaşlılık tarifi (denetimli klonlama) ya da Türkçe yazılmış tarif
İki aşamalı yapay ses (13.10): chatterbox adayı "metin" ile tek uzun Türkçe paragraf okur (00.flac); bu kayıt
voxcpm'e referans olur ve tarifle yaşlandırılır. Aksan Chatterbox'tan, ses rengi ve yaş VoxCPM2'den; gerçek kişi yok.
AYAR: {"motor": "chatterbox" | "voxcpm", "adaylar": {ad: {...}}}
  chatterbox aday: {"ref": "hasan6", "cfg": 0.0, "abarti": 0.5}
  voxcpm aday:     {"ref": "kuzey", "tarif": "...", "cfg": 2.0}  ya da  {"tarif": "..."} (tasarım: 00 kimlik olur)
Girdi (özel veri seti "seslen-dene"): ref/<ad>.wav, kadro.py, olcu.py. Çıktı: /kaggle/working/dene/<aday>/NN.flac + olcum.json
"""

import glob
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

ayar = globals().get("AYAR") or {}
GIRDI = Path(glob.glob("/kaggle/input/**/kadro.py", recursive=True)[0]).parent
CIKTI = Path("/kaggle/working/dene")
t_bas = time.time()
paket = ["chatterbox-tts"] if ayar["motor"] == "chatterbox" else ["voxcpm==2.0.3"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *paket, "faster-whisper==1.2.1", "soxr", "librosa"], check=True)
if ayar["motor"] == "chatterbox":     # chatterbox torch 2.6 kurar; Kaggle'ın torchvision'ı (torch 2.11 için) çakışıyor
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"], check=False)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(GIRDI))
import kadro  # noqa: E402
import olcu  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

gpu = torch.cuda.device_count()
olcu._whisper.cache_clear()
_w = WhisperModel("medium", device="cuda", device_index=1 if gpu > 1 else 0, compute_type="int8_float16")
olcu._whisper = lambda: _w


def ref(ad: str) -> str:
    return str(next(Path("/kaggle/input").rglob(f"ref/{ad}.wav")))


if ayar["motor"] == "chatterbox":
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda:0")
    sr = model.sr

    def uret(a: dict, metin: str, i: int, d: Path):
        torch.manual_seed(zlib.crc32(f"{a}|{i}".encode()))
        w = model.generate(metin, language_id="tr", audio_prompt_path=ref(a["ref"]), cfg_weight=a.get("cfg", 0.5),
                           exaggeration=a.get("abarti", 0.5))
        return w.squeeze().detach().cpu().numpy().astype(np.float32)
else:
    from huggingface_hub import snapshot_download
    from voxcpm import VoxCPM
    if torch.cuda.get_device_capability(0)[0] < 8:          # T4: bf16 yok → float32
        import voxcpm.model.voxcpm2 as _v2
        _v2.pick_runtime_dtype = lambda device, dtype: "float32"
    model = VoxCPM.from_pretrained(snapshot_download("openbmb/VoxCPM2", local_dir="/kaggle/tmp/VoxCPM2"),
                                   load_denoiser=False, optimize=False, device="cuda:0", local_files_only=True)
    sr = model.tts_model.sample_rate

    def uret(a: dict, metin: str, i: int, d: Path):
        torch.manual_seed(zlib.crc32(f"{a}|{i}".encode()))
        r = ref(a["ref"]) if a.get("ref") else (str(d / "00.flac") if i else None)   # tasarım: 00 kimlik olur
        k = {"reference_wav_path": r} if r else {}
        return model.generate(text=f"({a['tarif']}){metin}" if a.get("tarif") else metin, cfg_value=a.get("cfg", 2.0),
                              inference_timesteps=10, **k)
print(f"model hazır ({time.time() - t_bas:.0f} sn)", flush=True)

for ad, a in ayar["adaylar"].items():
    d = CIKTI / ad
    d.mkdir(parents=True, exist_ok=True)
    kayit = {"ayar": a, "olcum": []}
    for i, (duygu, cumle) in enumerate([("notr", a["metin"])] if a.get("metin") else kadro.DENEME):
        w = uret(a, cumle, i, d)
        sf.write(d / f"{i:02d}.flac", w, sr, subtype="PCM_16")
        dog, duy = olcu.dogruluk(w, sr, cumle)
        kayit["olcum"].append({"cumle": cumle, "dogruluk": round(dog, 3), "netlik": round(olcu.netlik(w, sr), 1),
                               "duyulan": duy, "sn": round(len(w) / sr, 2)})
        print(f"{ad} {i} doğ {dog:.2f} | {duy}", flush=True)
    (d / "olcum.json").write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"bitti: {len(ayar['adaylar'])} aday, {(time.time() - t_bas) / 60:.0f} dk", flush=True)
