# -*- coding: utf-8 -*-
"""
KAGGLE BETİKLERİ MODAL'DA (19.10) — Kaggle'ın haftalık 30 saatlik GPU kotası dolunca öğretmen işleri (VoxCPM2) Modal'ın
aylık hediye kredisiyle sürer. bulut/kaggle/ altındaki betikler (ogretmenDene.py, yeniden.py) DEĞİŞMEDEN koşar: kapsayıcıda
/kaggle/input/<veri seti> ve /kaggle/working hacme bağlanır, betiğin başına kaggleIs.it gibi AYAR satırı eklenir.

  python bulut/modal/ogretmenModal.py dene  --girdi cikti/kaggle/dene    --ayar ayar.json [--gpu A100]
  python bulut/modal/ogretmenModal.py yeniden --girdi cikti/kaggle/yeniden --ayar ayar.json --parca 6
Girdi klasörü hacme yüklenir (kaggleIs'in veri seti klasörüyle aynı içerik); çıktı
cikti/kaggle/indirilen/modal-<iş>/ altına iner (Kaggle çıktısıyla aynı düzen: dene/<aday>/…, veri/<ses>/…).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import modal

_YOL = Path(__file__).resolve()
KOK = _YOL.parents[2] if len(_YOL.parents) > 2 else _YOL.parent
KAGGLE = KOK / "bulut" / "kaggle"
SAAT_UCRET = {"A100": 2.10, "A10G": 1.10, "L4": 0.80, "H100": 3.95}

app = modal.App("seslen-ogretmen")
hacim = modal.Volume.from_name("seslen-ogretmen", create_if_missing=True)
# faster-whisper (ctranslate2) CUDA 12 + cuDNN 9 kitaplıklarını sistemde arıyor: debian_slim'de libcublas.so.12 yoktu
imaj = (modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
        .apt_install("libsndfile1", "ffmpeg", "git")
        .pip_install("voxcpm==2.0.3", "faster-whisper==1.2.1", "soxr", "librosa", "transformers", "huggingface_hub"))


@app.function(image=imaj, gpu="A100", volumes={"/vol": hacim}, timeout=6 * 3600)
def calistir(betik_adi: str, kod: str, girdi: str, is_adi: str) -> str:
    """Kaggle düzenini kurar, betiği çalıştırır; çıktı /vol/cikti/<is_adi>."""
    import shutil
    import subprocess
    # glob/rglob sembolik bağı her sürümde izlemiyor: girdi kopyalanır, çıktı düzenli olarak hacme kopyalanır
    shutil.copytree(f"/vol/girdi/{girdi}", f"/kaggle/input/{girdi}", dirs_exist_ok=True)
    for d in ("/kaggle/tmp", "/kaggle/working"):
        Path(d).mkdir(parents=True, exist_ok=True)
    hedef = f"/vol/cikti/{is_adi}"

    def kaydet():
        shutil.copytree("/kaggle/working", hedef, dirs_exist_ok=True)
        hacim.commit()
    Path(f"/root/{betik_adi}").write_text(kod, encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-u", f"/root/{betik_adi}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, cwd="/root")
    son, kayit = [], time.time()
    for satir in p.stdout:
        print(satir, end="", flush=True)
        son = (son + [satir])[-40:]
        if time.time() - kayit > 600:
            kaydet()
            kayit = time.time()
    p.wait()
    kaydet()
    return f"çıkış {p.returncode}\n" + "".join(son)


def _yukle(girdi: Path, ad: str) -> None:
    with hacim.batch_upload(force=True) as b:
        for f in girdi.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                b.put_file(str(f), f"girdi/{ad}/{f.relative_to(girdi).as_posix()}")


def _indir(is_adi: str) -> Path:
    hedef = KOK / "cikti" / "kaggle" / "indirilen" / f"modal-{is_adi}"
    say = 0
    for e in hacim.iterdir(f"cikti/{is_adi}", recursive=True):
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        yer = hedef / Path(e.path).relative_to(f"cikti/{is_adi}")
        yer.parent.mkdir(parents=True, exist_ok=True)
        with open(yer, "wb") as f:
            for parca in hacim.read_file(e.path):
                f.write(parca)
        say += 1
    print(f"indirildi: {say} dosya → {hedef}", flush=True)
    return hedef


def is_calistir(betik: str, girdi: Path, ayar: dict, is_adi: str, gpu: str) -> None:
    ad = girdi.name if girdi.name != "dene" else "seslen-dene"
    print(f"girdi yükleniyor: {girdi} → {ad}", flush=True)
    _yukle(girdi, ad)
    kod = f"AYAR = {ayar!r}\n" + (KAGGLE / betik).read_text(encoding="utf-8")
    t0 = time.time()
    with modal.enable_output(), app.run():
        print(calistir.with_options(gpu=gpu).remote(betik, kod, ad, is_adi)[-3000:], flush=True)
    sa = (time.time() - t0) / 3600
    print(f"süre {sa:.2f} sa ≈ {sa * SAAT_UCRET.get(gpu, 2.1):.2f} $ kredi", flush=True)
    _indir(is_adi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("is_", choices=["dene", "yeniden"])
    ap.add_argument("--girdi", required=True, help="kaggleIs'in veri seti klasörü (cikti/kaggle/dene, …/yeniden)")
    ap.add_argument("--ayar", required=True, help="AYAR sözlüğünün JSON dosyası")
    ap.add_argument("--parca", type=int, default=0, help="yeniden: plan<N>.jsonl")
    ap.add_argument("--ad", default="", help="iş adı (çıktı klasörü)")
    ap.add_argument("--gpu", default="A100")
    a = ap.parse_args()
    ayar = json.loads(Path(a.ayar).read_text(encoding="utf-8"))
    if a.is_ == "yeniden":
        ayar["parca"] = a.parca
    betik = {"dene": "ogretmenDene.py", "yeniden": "yeniden.py"}[a.is_]
    is_calistir(betik, KOK / a.girdi, ayar, a.ad or f"{a.is_}-{a.parca or int(time.time())}", a.gpu)
