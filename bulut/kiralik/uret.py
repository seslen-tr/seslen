# -*- coding: utf-8 -*-
"""
KİRALIK GPU ÜRETİMİ — kiralık makinede (RunPod) çalışır; bulut/kiralik/runpod.py gönderir.

    python uret.py --paket /workspace/paket --cikti /workspace/cikti --sesler ayse,serdar [--aday 2]

Paket: plan/<ses>.jsonl (plan.py), ref/<ses>.wav (tazelenmiş kimlik, ref2),
olcu.py, duygu.py, kadro.py. Her cümle plandaki kiple okunur:
  B  kimlik referans + duygu tarifi ("(tarif)metin")
  C  kimlik referans + devam kipi (prompt = kimlik + TAZELE_METIN) — yalnız nötr
Aday: en çok --aday okuma; ilki çok iyiyse (doğ ≥ 0,97, net ≥ 21) ikincisi
okunmaz; hiçbiri kapıdan (0,90 / 18) geçmezse bir okuma daha. En iyisi kaydedilir.
Kaldığı yerden sürer (olcum.jsonl'deki kimlikler atlanır). ilerleme.json her
cümlede güncellenir; bitince BITTI yazılır.
bf16: 4090 / A-serisi (sm80+) donanımda destekler — VoxCPM2 varsayılanı.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--paket", default="/workspace/paket")
ap.add_argument("--cikti", default="/workspace/cikti")
ap.add_argument("--sesler", required=True)
ap.add_argument("--aday", type=int, default=2)
ap.add_argument("--adim", type=int, default=10)
ap.add_argument("--sinir", type=int, default=0, help="ses başına en çok bu kadar cümle (deneme)")
ap.add_argument("--optimize", action="store_true", help="VoxCPM torch.compile (GPU'da hızlı, ilk derleme ~dakikalar)")
a = ap.parse_args()
PAKET, CIKTI = Path(a.paket), Path(a.cikti)
sys.path.insert(0, str(PAKET))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import duygu  # noqa: E402
import kadro  # noqa: E402
import olcu  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

olcu._whisper.cache_clear()
_w = WhisperModel("medium", device="cuda", compute_type="int8_float16")
olcu._whisper = lambda: _w
from voxcpm import VoxCPM  # noqa: E402

t_bas = time.time()
model = VoxCPM.from_pretrained("/workspace/VoxCPM2", load_denoiser=False, optimize=a.optimize, device="cuda:0",
                               local_files_only=True)
sr = model.tts_model.sample_rate
print(f"model hazır ({time.time() - t_bas:.0f} sn), dtype {model.tts_model.config.dtype}", flush=True)
KAPI, COK_IYI = (0.90, 18.0), (0.97, 21.0)

sesler = [s for s in a.sesler.split(",") if s]
planlar = {s: [json.loads(x) for x in (PAKET / "plan" / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
           for s in sesler}
if a.sinir:
    planlar = {s: p[: a.sinir] for s, p in planlar.items()}
toplam = sum(len(p) for p in planlar.values())
ilerleme = {"toplam": toplam, "biten": 0, "gecen": 0, "ses_sn": 0.0, "is_sn": 0.0, "basla": time.time()}


def puanla(wav, metin):
    d, duy = olcu.dogruluk(wav, sr, metin)
    n = olcu.netlik(wav, sr)
    return {"dogruluk": round(d, 3), "netlik": round(n, 1), "duyulan": duy,
            "gecti": d >= KAPI[0] and n >= KAPI[1], "cok_iyi": d >= COK_IYI[0] and n >= COK_IYI[1]}


def oku(ses, c, i):
    ref = str(PAKET / "ref" / f"{ses}.wav")
    torch.manual_seed(zlib.crc32(f"{ses}|{c['kimlik']}|{i}".encode()))
    if c["kip"] == "C":
        return model.generate(text=c["metin"], reference_wav_path=ref, prompt_wav_path=ref,
                              prompt_text=kadro.TAZELE_METIN, cfg_value=2.0, inference_timesteps=a.adim)
    return model.generate(text=duygu.metne_ekle(c["duygu"], c["metin"]), reference_wav_path=ref,
                          cfg_value=c.get("cfg", 2.0), inference_timesteps=a.adim)    # cfg: plan satırı seçebilir


for ses, plan in planlar.items():
    d = CIKTI / ses
    (d / "wavs").mkdir(parents=True, exist_ok=True)
    olcum = d / "olcum.jsonl"
    biten = {json.loads(x)["kimlik"] for x in olcum.read_text(encoding="utf-8").splitlines()} if olcum.exists() else set()
    ilerleme["biten"] += len(biten & {c["kimlik"] for c in plan})
    for c in plan:
        if c["kimlik"] in biten:
            continue
        t0 = time.perf_counter()
        adaylar = []
        for i in range(a.aday + 1):
            if adaylar and (adaylar[-1][0]["cok_iyi"] or (i == a.aday and any(x[0]["gecti"] for x in adaylar))):
                break
            wav = oku(ses, c, i)
            adaylar.append((puanla(wav, c["metin"]), wav))
        p, wav = max(adaylar, key=lambda x: (x[0]["gecti"], x[0]["dogruluk"], x[0]["netlik"]))
        sf.write(d / "wavs" / f"{c['kimlik']}.flac", wav, sr, subtype="PCM_16")
        sure = time.perf_counter() - t0
        o = {**c, "sn": round(len(wav) / sr, 2), "netlik": p["netlik"], "dogruluk": p["dogruluk"],
             "duyulan": p["duyulan"], "gecti": p["gecti"], "deneme": len(adaylar), "uretim_sn": round(sure, 1),
             "adaylar": [{"dogruluk": x[0]["dogruluk"], "netlik": x[0]["netlik"]} for x in adaylar]}
        with open(olcum, "a", encoding="utf-8") as f:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
        ilerleme["biten"] += 1
        ilerleme["gecen"] += p["gecti"]
        ilerleme["ses_sn"] += o["sn"]
        ilerleme["is_sn"] += sure
        (CIKTI / f"ilerleme-{a.sesler.replace(',', '_')}.json").write_text(json.dumps({**ilerleme, "ses": ses, "an": time.time()}), encoding="utf-8")
        print(f"{ses} {c['kimlik']} {c['duygu']:7s} {c['kip']} {'✓' if p['gecti'] else '✗'} {p['dogruluk']:.2f} "
              f"{p['netlik']:.1f} ({sure:.1f} sn, {len(adaylar)} okuma, ses {o['sn']} sn)", flush=True)
r = ilerleme["is_sn"] / max(ilerleme["ses_sn"], 1)
print(f"ÖZET: {ilerleme['ses_sn'] / 3600:.2f} sa ses, {ilerleme['is_sn'] / 3600:.2f} sa iş, RTF {r:.2f}", flush=True)
(CIKTI / f"BITTI-{a.sesler.replace(',', '_')}").write_text(json.dumps(ilerleme), encoding="utf-8")
