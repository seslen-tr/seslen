# -*- coding: utf-8 -*-
"""
KAGGLE ÜRETİMİ — öğretmen verisi (VoxCPM2) Kaggle'ın ücretsiz GPU'sunda.

Bu dosya Kaggle'da "script kernel" olarak koşar (bulut/kaggle/gonder.py
gönderir). Girdi: özel Kaggle veri seti "sesatolye-girdi" (paketle.py yapar):
    korpus.jsonl, ref/<karakter>.wav, olcu.py, ayar.json
Çıktı: /kaggle/working/veri/<karakter>/wavs/<kimlik>.flac + olcum.jsonl
(indir.py bilgisayara çeker, cikti/veri/ ile birleştirir).

ayar.json: {"bas": 0, "son": 20, "adim": 6, "sesler": ["damla", "kuzey"],
            "sure_dk": 690}
İki GPU varsa (T4×2) VoxCPM2 birinde, Whisper öbüründe koşar.
Kapı yereldekiyle aynı (ogretmen/veriUret.py): doğruluk ≥ 0,90, netlik ≥ 18.
"""

import glob
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

GIRDI = Path(glob.glob("/kaggle/input/**/korpus.jsonl", recursive=True)[0]).parent


def ref_yolu(kar: str) -> Path:
    """ref/<kar>.wav — Kaggle klasörü açmış olabilir ya da ref.zip olarak bırakmış."""
    bul = glob.glob(f"/kaggle/input/**/{kar}.wav", recursive=True)
    if bul:
        return Path(bul[0])
    import zipfile
    hedef = Path("/kaggle/tmp/ref")
    if not (hedef / f"{kar}.wav").exists():
        zipfile.ZipFile(next(GIRDI.glob("ref*.zip"))).extractall(hedef)
    return next(hedef.rglob(f"{kar}.wav"))
CIKTI = Path("/kaggle/working/veri")
t_bas = time.time()


def kur() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "voxcpm==2.0.3", "faster-whisper==1.2.1",
                    "soxr", "librosa"], check=True)


kur()
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

# AYAR: kaggleIs.py gonder bu dosyanın başına "AYAR = {...}" yazar.
ayar = globals().get("AYAR") or json.loads((GIRDI / "ayar.json").read_text(encoding="utf-8"))
gpu_sayisi = torch.cuda.device_count()
print(f"GPU: {[torch.cuda.get_device_name(i) for i in range(gpu_sayisi)]}", flush=True)

# Whisper ikinci GPU'da (varsa): olcu._whisper cuda'yı seçer; CUDA_VISIBLE
# ayarı süreç başında yapılamadığı için cihaz dizinini yamayla veriyoruz.
sys.path.insert(0, str(GIRDI))
import olcu  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

olcu._whisper.cache_clear()
_w = WhisperModel("medium", device="cuda", device_index=1 if gpu_sayisi > 1 else 0,
                  compute_type="int8_float16")
olcu._whisper = lambda: _w

from huggingface_hub import snapshot_download  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

# bf16 yalnız Ampere+ (sm80) GPU'da donanımda; Kaggle'ın T4 (sm75) ve P100
# (sm60) kartlarında yok. fp16 da riskli: VoxCPM kendi notunda düşük
# duyarlıkta difüzyon döngüsünün bozulduğunu yazıyor (MPS). → float32
# (≈ 9 GB, 16 GB karta sığar).
if torch.cuda.get_device_capability(0)[0] < 8:
    import voxcpm.model.voxcpm2 as _v2
    _v2.pick_runtime_dtype = lambda device, dtype: "float32"

model_yolu = snapshot_download("openbmb/VoxCPM2", local_dir="/kaggle/tmp/VoxCPM2")
model = VoxCPM.from_pretrained(model_yolu, load_denoiser=False, optimize=False, device="cuda:0",
                               local_files_only=True)
sr = model.tts_model.sample_rate
print(f"model hazır ({time.time() - t_bas:.0f} sn), dtype {model.tts_model.config.dtype}", flush=True)

import duygu as duygu_  # noqa: E402  (girdi veri setinde; ogretmen/duygu.py)

korpus = [json.loads(s) for s in (GIRDI / "korpus.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
tum = korpus
korpus = korpus[ayar["bas"]:ayar["son"]]
# MERAK EKİ: soruların çoğu derlemin ilk yarısında (korpusYap sırası) ve o
# yarı etiketsiz okundu. İstenirse oradan N soru "merak" ile YENİDEN okunur
# (kimlik sonuna "m"): öğrenci soru tonlamasını da duysun.
if ayar.get("merak_ek"):
    sorular = [c for c in tum[:ayar["bas"]] if c["metin"].endswith("?")][: ayar["merak_ek"]]
    korpus += [dict(c, kimlik=c["kimlik"] + "m", duygu="merak") for c in sorular]
bitis = t_bas + ayar.get("sure_dk", 690) * 60
ADAY = int(ayar.get("aday", 1))
KAPI = (0.90, 18.0)
top_ses = top_is = 0.0


def puanla(wav, metin):
    dog, duy = olcu.dogruluk(wav, sr, metin)
    net = olcu.netlik(wav, sr)
    return (dog >= KAPI[0] and net >= KAPI[1], round(dog, 3), round(net, 1)), duy


for c in korpus:
    if time.time() > bitis:
        print("süre doldu, duruluyor", flush=True)
        break
    duygu = c.get("duygu") or (duygu_.sec(c["kimlik"], c["metin"]) if ayar.get("duygu") else "notr")
    metin = duygu_.metne_ekle(duygu, c["metin"])
    for kar in ayar["sesler"]:
        d = CIKTI / kar
        (d / "wavs").mkdir(parents=True, exist_ok=True)
        ref = ref_yolu(kar)
        t0 = time.perf_counter()
        adaylar = []       # (puan, duyulan, wav)
        # EN İYİDEN ÖĞRENME: ADAY kadar okuma, en iyisi eğitime; hiçbiri kapıdan
        # geçmezse bir okuma daha. Seçilmeyenler de saklanır (".aN") —
        # ileride zevk modeli / oylama için.
        for i in range(ADAY + 1):
            if i == ADAY and any(a[0][0] for a in adaylar):
                break
            torch.manual_seed(zlib.crc32(f"{c['kimlik']}|{kar}|{i}".encode()))
            wav = model.generate(text=metin, reference_wav_path=str(ref), cfg_value=2.0,
                                 inference_timesteps=ayar.get("adim", 6))
            puan, duy = puanla(wav, c["metin"])
            adaylar.append((puan, duy, wav))
        sirali = sorted(range(len(adaylar)), key=lambda k: adaylar[k][0], reverse=True)
        (gecti, dog, net), duy, wav = adaylar[sirali[0]]
        sf.write(d / "wavs" / f"{c['kimlik']}.flac", wav, sr, subtype="PCM_16")
        for sira, k in enumerate(sirali[1:], 1):
            sf.write(d / "wavs" / f"{c['kimlik']}.a{sira}.flac", adaylar[k][2], sr, subtype="PCM_16")
        sure = time.perf_counter() - t0
        o = {"kimlik": c["kimlik"], "metin": c["metin"], "kaynak": c["kaynak"], "duygu": duygu,
             "sn": round(len(wav) / sr, 2), "netlik": net, "dogruluk": dog, "duyulan": duy, "gecti": gecti,
             "deneme": len(adaylar), "uretim_sn": round(sure, 1),
             "adaylar": [{"dogruluk": adaylar[k][0][1], "netlik": adaylar[k][0][2]} for k in sirali]}
        with open(d / "olcum.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
        top_ses += o["sn"]
        top_is += sure
        print(f"{kar} {c['kimlik']} {duygu:7s} {'✓' if gecti else '✗'} {dog:.2f} {net:.1f} "
              f"({sure:.1f} sn, {len(adaylar)} okuma, ses {o['sn']} sn)", flush=True)
if top_ses:
    print(f"ÖZET: {top_ses / 60:.1f} dk ses, {top_is / 60:.1f} dk iş, RTF {top_is / top_ses:.2f}", flush=True)
