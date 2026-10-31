# -*- coding: utf-8 -*-
"""
KAGGLE YENİDEN ÜRETİM — plandaki cümleleri sesin kimlik kaydıyla öğretmene (VoxCPM2) okutur. Script kernel olarak
koşar; kaggleIs.py yeniden gönderir (bulut/kiralik/yenile.py plan + paketle yazar). 08.10'da Serdar'ın güçlü tarifli
duyguları, 10.10'da Fatma'nın yeni sesi (fatma-5) ve bütün seslerin kısa cümleleri bununla üretildi.

Girdi (özel veri seti "seslen-yeniden"):
  plan<N>.jsonl   {"ses", "kimlik", "metin", "duygu", "kip": A/B/C, "cfg"?}; AYAR["parca"] = N
  ref/<ses>.wav   tasarım kaydı (~4 sn);  ref/<ses>2.wav tazelenmiş kimlik (~10 sn), yoksa burada üretilir
  olcu.py, duygu.py, kadro.py
Kipler (tazele.py ile aynı): A ref.wav + adım 6; B ref2 + adım 10; C ref2 + devam kipi (ref2 + TAZELE_METIN) + adım 10.
  D (15.10, "1. yol"): ref.wav Chatterbox'un Türkçe paragrafı + sesin karakter tarifi AYAR["tarif"][ses] (duygu tarifiyle
    birlikte, adım 10); kimlik kaydı ref2 aynı yöntemle TAZELE_METIN'den üretilir (indirilsin diye).
Çıktı: /kaggle/working/veri/<ses>/wavs/<kimlik>.flac + olcum.jsonl (uret.py biçimi); üretilen ref2 → ref/<ses>2.wav.
Aday: en çok AYAR["aday"] okuma (ilki çok iyiyse ikincisi yok), hiçbiri kapıdan geçmezse bir okuma daha.
Kapı: doğruluk ≥ 0,90, netlik ≥ 18; hırıltılı tasarım seslerde (fatma-5) netlik eşiği AYAR["kapi_netlik"][ses].
16.10: + harf harf denetim (araclar/olcu/denetim.py: CTC telaffuz puanı, başta/sonda fazladan ses). Whisper bozuk
sözcüğü dil modeliyle doğru yazıyordu; ilk 6 seste kayıtların %19-36'sı denetime takıldı. Plan satırında "ipucu"
(konuşma eylemi: hitap, yasak, soru, masal; yenile.py turkce-duygu ile yazar) tarife eklenir.
"""

import glob
import json
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

ayar = globals().get("AYAR") or {}
PARCA = int(ayar.get("parca", 1))
GIRDI = Path(glob.glob(f"/kaggle/input/**/plan{PARCA}.jsonl", recursive=True)[0]).parent
CIKTI = Path("/kaggle/working")
t_bas = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "voxcpm==2.0.3", "faster-whisper==1.2.1", "soxr",
                "librosa", "transformers"], check=True)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

gpu = torch.cuda.device_count()
print(f"GPU: {[torch.cuda.get_device_name(i) for i in range(gpu)]}", flush=True)
sys.path.insert(0, str(GIRDI))
import duygu  # noqa: E402
import kadro  # noqa: E402
import olcu  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

olcu._whisper.cache_clear()
_w = WhisperModel("medium", device="cuda", device_index=1 if gpu > 1 else 0, compute_type="int8_float16")
olcu._whisper = lambda: _w
import denetim  # noqa: E402

denetim.CIHAZ = "cuda:1" if gpu > 1 else "cuda:0"

from huggingface_hub import snapshot_download  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

if torch.cuda.get_device_capability(0)[0] < 8:          # T4 / P100: bf16 yok → float32 (≈ 9 GB)
    import voxcpm.model.voxcpm2 as _v2
    _v2.pick_runtime_dtype = lambda device, dtype: "float32"
model = VoxCPM.from_pretrained(snapshot_download("openbmb/VoxCPM2", local_dir="/kaggle/tmp/VoxCPM2"),
                               load_denoiser=False, optimize=False, device="cuda:0", local_files_only=True)
sr = model.tts_model.sample_rate
print(f"model hazır ({time.time() - t_bas:.0f} sn)", flush=True)

COK_IYI = (0.97, 21.0)
ADAY, ADIM_A = int(ayar.get("aday", 2)), 6
bitis = t_bas + ayar.get("sure_dk", 690) * 60
plan = [json.loads(s) for s in (GIRDI / f"plan{PARCA}.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
(CIKTI / "ref").mkdir(parents=True, exist_ok=True)


def kapi(ses: str) -> tuple[float, float]:
    return 0.90, float(ayar.get("kapi_netlik", {}).get(ses, 18.0))


def olc(wav, metin: str, ses: str) -> dict:
    d, duy = olcu.dogruluk(wav, sr, metin)
    n = olcu.netlik(wav, sr)
    k = kapi(ses)
    dn = denetim.denetle(wav, sr, metin)
    temiz = not dn["bayrak"]
    return {"dogruluk": round(d, 3), "netlik": round(n, 1), "duyulan": duy, "gop": dn["gop"], "denetim": dn["neden"],
            "gecti": d >= k[0] and n >= k[1] and temiz,
            "cok_iyi": temiz and d >= COK_IYI[0] and n >= max(k[1], COK_IYI[1] - (18.0 - k[1]))}


def kimlik(ses: str) -> tuple[str, str]:
    """(ref.wav, ref2.wav) yolları; ref2 yoksa tazele.py'deki gibi 3 tohumla üretilir, kapıdan geçen en neti."""
    ref = str(next(Path("/kaggle/input").rglob(f"ref/{ses}.wav")))
    hazir = list(Path("/kaggle/input").rglob(f"ref/{ses}2.wav"))
    if hazir:
        return ref, str(hazir[0])
    hedef = CIKTI / "ref" / f"{ses}2.wav"
    if not hedef.exists():
        adaylar = []
        for i in range(3):
            torch.manual_seed(zlib.crc32(f"{ses}|tazele|{i}".encode()))
            wav = model.generate(text=kadro.TAZELE_METIN, reference_wav_path=ref, cfg_value=2.0, inference_timesteps=10)
            adaylar.append((olc(wav, kadro.TAZELE_METIN, ses), wav))
        o, wav = max(adaylar, key=lambda a: (a[0]["gecti"], a[0]["netlik"]))
        sf.write(hedef, wav, sr)
        print(f"{ses}: ref2 üretildi, doğ {o['dogruluk']:.2f} net {o['netlik']} (adaylar "
              f"{[a[0]['netlik'] for a in adaylar]})", flush=True)
    return ref, str(hedef)


TARIF_SES = ayar.get("tarif", {})


def yol1_metin(ses: str, duygu_ad: str, metin: str, ipucu: str = "") -> str:
    tarif = ", ".join(x for x in (TARIF_SES[ses], duygu.TARIF[duygu_ad], ipucu) if x)
    return f"({tarif}){metin}"


def yol1_kimlik(ses: str) -> tuple[str, None]:
    """D kipi: (ref.wav, None); kimlik kaydı ref/<ses>2.wav aynı yöntemle (3 tohum, kapıdan geçen en neti)."""
    ref = str(next(Path("/kaggle/input").rglob(f"ref/{ses}.wav")))
    hedef = CIKTI / "ref" / f"{ses}2.wav"
    if not hedef.exists():
        adaylar = []
        for i in range(3):
            torch.manual_seed(zlib.crc32(f"{ses}|yol1|{i}".encode()))
            wav = model.generate(text=yol1_metin(ses, "notr", kadro.TAZELE_METIN), reference_wav_path=ref,
                                 cfg_value=2.0, inference_timesteps=10)
            adaylar.append((olc(wav, kadro.TAZELE_METIN, ses), wav))
        o, wav = max(adaylar, key=lambda a: (a[0]["gecti"], a[0]["netlik"]))
        sf.write(hedef, wav, sr)
        print(f"{ses}: 1. yol kimlik kaydı, doğ {o['dogruluk']:.2f} net {o['netlik']}", flush=True)
    return ref, None


refler: dict[str, tuple[str, str | None]] = {}
top_ses = top_is = 0.0
for c in plan:
    if time.time() > bitis:
        print("süre doldu, duruluyor", flush=True)
        break
    ses = c["ses"]
    yol1 = c.get("kip") == "D"
    if ses not in refler:
        refler[ses] = yol1_kimlik(ses) if yol1 else kimlik(ses)
    ref, ref2 = refler[ses]
    kip = {"A": lambda: dict(reference_wav_path=ref, inference_timesteps=ADIM_A),
           "B": lambda: dict(reference_wav_path=ref2, inference_timesteps=10),
           "C": lambda: dict(reference_wav_path=ref2, prompt_wav_path=ref2, prompt_text=kadro.TAZELE_METIN,
                             inference_timesteps=10),
           "D": lambda: dict(reference_wav_path=ref, inference_timesteps=10)}[c.get("kip", "B")]()
    metin = yol1_metin(ses, c["duygu"], c["metin"], c.get("ipucu", "")) if yol1 else duygu.metne_ekle(c["duygu"], c["metin"])
    t0 = time.perf_counter()
    adaylar = []
    for i in range(ADAY + 1):
        if adaylar and (adaylar[-1][0]["cok_iyi"] or (i == ADAY and any(a[0]["gecti"] for a in adaylar))):
            break
        torch.manual_seed(zlib.crc32(f"{ses}|{c['kimlik']}|{i}".encode()))
        wav = model.generate(text=metin, cfg_value=c.get("cfg", 2.0), **kip)
        adaylar.append((olc(wav, c["metin"], ses), wav))
    p, wav = max(adaylar, key=lambda x: (x[0]["gecti"], not x[0]["denetim"], x[0]["gop"], x[0]["dogruluk"], x[0]["netlik"]))
    d = CIKTI / "veri" / ses
    (d / "wavs").mkdir(parents=True, exist_ok=True)
    sf.write(d / "wavs" / f"{c['kimlik']}.flac", wav, sr, subtype="PCM_16")
    sure = time.perf_counter() - t0
    o = {**c, "sn": round(len(wav) / sr, 2), "netlik": p["netlik"], "dogruluk": p["dogruluk"], "duyulan": p["duyulan"],
         "gop": p["gop"], "denetim": p["denetim"],
         "gecti": p["gecti"], "deneme": len(adaylar), "uretim_sn": round(sure, 1), "yer": "kaggle",
         "adaylar": [{"dogruluk": x[0]["dogruluk"], "netlik": x[0]["netlik"], "gop": x[0]["gop"]} for x in adaylar]}
    with open(d / "olcum.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
    top_ses += o["sn"]
    top_is += sure
    print(f"{ses} {c['kimlik']} {c['duygu']} {'✓' if p['gecti'] else '✗'} {p['dogruluk']:.2f} {p['netlik']:.1f} "
          f"({sure:.1f} sn, {len(adaylar)} okuma)", flush=True)
for s in refler:                                       # kullanılan kimlik kayıtları da indirilsin
    if not (CIKTI / "ref" / f"{s}2.wav").exists() and refler[s][1]:
        shutil.copy(refler[s][1], CIKTI / "ref" / f"{s}2.wav")
if top_ses:
    print(f"ÖZET: {top_ses / 60:.1f} dk ses, {top_is / 60:.1f} dk iş, RTF {top_is / top_ses:.2f}", flush=True)
