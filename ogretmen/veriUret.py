# -*- coding: utf-8 -*-
"""
ÖĞRETMEN VERİSİ — seçilen seslerle derlemi (veri/korpus.jsonl) okutur.

Kaldığı yerden sürer: var olan kayıt atlanır; istediğin an durdur.
Her cümle sırayla BÜTÜN karakterlerle okunur (ikisi eşit ilerlesin).
Her kayıt ölçülür; KAPI: Whisper doğruluğu ≥ 0,90 ve netlik ≥ 18 dB.
Geçemeyen kayıt bir kez başka tohumla yeniden üretilir, iyisi kalır;
yine geçemezse "gecti": false işaretlenir (eğitime girmez, silinmez).

    python ogretmen/veriUret.py --sure-dk 480          # gece: 8 saat
    python ogretmen/veriUret.py --adet 5               # deneme

Çıkış: cikti/veri/<karakter>/wavs/<kimlik>.wav (48 kHz) + olcum.jsonl
       + metadata.csv (yalnız geçenler; "kimlik|metin", piper/LJSpeech biçimi)
Makine uyumaz (SetThreadExecutionState), EcoQoS kapalı (yukle.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "araclar" / "olcu"))
from olcu import dogruluk, netlik  # noqa: E402
from yukle import KOK, ogretmen  # noqa: E402

# Seçim 28.09.2026 (dinleti): Damla = damla-4, Kuzey = kuzey-1.
SESLER = {"damla": "damla-4", "kuzey": "kuzey-1"}
KAPI_DOGRULUK, KAPI_NETLIK = 0.90, 18.0
# hizSecenek.py (git geçmişinde; damla-4, 3 cümle, RTF = üretim / ses süresi):
#   A referans + devam kaydı, 10 adım  RTF 18,0  doğruluk 1,00
#   B yalnız referans, 10 adım         RTF 11,4  doğruluk 0,95
#   C yalnız referans, 6 adım          RTF  7,3  doğruluk 1,00  ← seçildi
AYAR = {"adim": 6, "devam": False}


def uyanik_tut() -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED


def korpus() -> list[dict]:
    yol = KOK / "veri" / "korpus.jsonl"
    return [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines() if s.strip()]


def metadata_yaz(karakter: str) -> int:
    d = KOK / "cikti" / "veri" / karakter
    satir = []
    for s in (d / "olcum.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(s)
        if o["gecti"]:
            satir.append(f"{o['kimlik']}|{o['metin']}")
    (d / "metadata.csv").write_text("\n".join(satir) + "\n", encoding="utf-8")
    return len(satir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adet", type=int, default=0, help="en çok bu kadar yeni cümle (0 = sınırsız)")
    ap.add_argument("--sure-dk", type=float, default=0, help="bu kadar dakika sonra dur (0 = sınırsız)")
    ap.add_argument("--karakter", default="", help="yalnız bu karakter (varsayılan: hepsi)")
    ap.add_argument("--cihaz", default="cpu")
    a = ap.parse_args()
    karakterler = [a.karakter] if a.karakter else list(SESLER)

    uyanik_tut()
    model = ogretmen(a.cihaz)
    sr = model.tts_model.sample_rate
    adaylar = json.loads((KOK / "ogretmen" / "adayMetin.json").read_text(encoding="utf-8"))
    bitis = time.time() + a.sure_dk * 60 if a.sure_dk else float("inf")
    yeni = 0
    top_ses = top_sure = 0.0
    for c in korpus():
        if time.time() > bitis or (a.adet and yeni >= a.adet):
            break
        yapildi = False
        for kar in karakterler:
            d = KOK / "cikti" / "veri" / kar
            (d / "wavs").mkdir(parents=True, exist_ok=True)
            hedef = d / "wavs" / f"{c['kimlik']}.wav"
            if hedef.exists():
                continue
            ref = KOK / "cikti" / "adaylar" / SESLER[kar] / "ref.wav"
            ek = dict(prompt_wav_path=str(ref), prompt_text=adaylar["ref_metin"]) if AYAR["devam"] else {}
            en_iyi = None
            t0 = time.perf_counter()
            for deneme in range(2):
                # crc32: hash() her süreçte farklı; tohum yeniden üretilebilir olsun
                torch.manual_seed(zlib.crc32(f"{c['kimlik']}|{kar}|{deneme}".encode()))
                wav = model.generate(text=c["metin"], reference_wav_path=str(ref), cfg_value=2.0,
                                     inference_timesteps=AYAR["adim"], **ek)
                dog, duy = dogruluk(wav, sr, c["metin"])
                net = netlik(wav, sr)
                puan = (dog >= KAPI_DOGRULUK and net >= KAPI_NETLIK, dog, net)
                if en_iyi is None or puan > en_iyi[0]:
                    en_iyi = (puan, wav, dog, duy, net)
                if puan[0]:
                    break
            (gecti, _, _), wav, dog, duy, net = en_iyi
            sure = time.perf_counter() - t0
            sf.write(hedef, wav, sr, subtype="PCM_16")
            o = {"kimlik": c["kimlik"], "metin": c["metin"], "kaynak": c["kaynak"], "sn": round(len(wav) / sr, 2),
                 "netlik": round(net, 1), "dogruluk": round(dog, 3), "duyulan": duy, "gecti": gecti,
                 "deneme": deneme + 1, "uretim_sn": round(sure, 1)}
            with open(d / "olcum.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
            top_ses += o["sn"]
            top_sure += sure
            yapildi = True
            print(f"{kar} {c['kimlik']} {'✓' if gecti else '✗'} doğruluk {dog:.2f} netlik {net:.1f} "
                  f"({sure:.0f} sn) | {c['metin']}", flush=True)
        yeni += yapildi
    for kar in karakterler:
        if (KOK / "cikti" / "veri" / kar / "olcum.jsonl").exists():
            print(f"{kar}: metadata {metadata_yaz(kar)} kayıt")
    if top_ses:
        print(f"bu koşu: {top_ses / 60:.1f} dk ses, {top_sure / 60:.1f} dk iş (RTF {top_sure / top_ses:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
