# -*- coding: utf-8 -*-
"""
3. AŞAMA YENİ MAKİNEDE — eski eğitim makinesinin sunucusunda GPU boşalmadığı için (04.10) 3. aşama
(vurgulu okunuş + Türkçe PL-BERT + 9 ton) sıfırdan açılan bir makinede kurulur ve başlatılır.

Yerelde gerekenler: veri/egitim (ogrenci/veri/veriHazirla.py), veri/egitim_vurgu (ogrenci/veri/vurguListe.py),
cikti/ogrenci/epoch_2nd_00004.pth (2. aşamanın son ağırlığı; 3. aşama yalnız parametrelerini alır),
turkce-plbert/model (Türkçe PL-BERT). Adımlar (her biri kaldığı yerden sürdürülebilir):
  ac       makineyi aç (rol "egitim"; yalnız 4090 / L40S: kalıptaki torch 2.4 5090'da çalışmıyor)
  kur      StyleTTS2 + paketler (egit.sh'in kurulum bölümü) + paket_egit + asama23.sh
  yukle    veri/egitim, egitim_vurgu, epoch_2nd_00004.pth, Türkçe PL-BERT (UC_HAZIR)
  baslat   asama23.sh: 2. aşama ağırlığı hazır olduğundan doğrudan 3. aşamaya geçer

    python bulut/kiralik/asama3_yeni.py hepsi      (ya da tek adım: ac | kur | yukle | baslat)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod as r  # noqa: E402

KOK = r.KOK
KURULUM = r"""
set -e
cd /workspace
pip install -q munch einops einops-exts pydub nltk "accelerate<1.3" "transformers==4.46.3" soundfile librosa pandas tensorboard matplotlib pyyaml cython
pip install -q --no-build-isolation git+https://github.com/resemble-ai/monotonic_align.git
[ -d StyleTTS2 ] || git clone -q --depth 1 https://github.com/yl4579/StyleTTS2
python -c "from huggingface_hub import hf_hub_download as h; h('yl4579/StyleTTS2-LJSpeech', 'Models/LJSpeech/epoch_2nd_00100.pth', local_dir='/workspace/StyleTTS2'); h('yl4579/StyleTTS2-LJSpeech', 'Models/LJSpeech/config.yml', local_dir='/workspace/StyleTTS2')"
touch /workspace/.egit_kuruldu
echo KURULDU
"""


def pod() -> dict:
    return next(x for x in r.podlar() if x.get("rol") == "egitim")


def ac() -> None:
    pl = r.podlar()
    for x in pl:
        if x["ad"] == "sesatolye-1" and not x.get("rol"):
            x["rol"] = "egitim-eski"          # GPU'su alınamayan eski makine (diski duruyor)
    r.podlar_yaz(pl)
    r.GPULAR[:] = ["NVIDIA GeForce RTX 4090", "NVIDIA L40S"]
    r.ac(1, guvenli=True, rol="egitim")


def gonder(yerel: list[tuple[Path, str]], hedef_kok: str) -> None:
    s = subprocess.Popen(["ssh", *r.ssh_arg(pod()), f"root@{pod()['ip']}", f"mkdir -p {hedef_kok} && tar xf - -C {hedef_kok}"],
                         stdin=subprocess.PIPE)
    with tarfile.open(fileobj=s.stdin, mode="w|") as t:
        for p, ad in yerel:
            t.add(p, arcname=ad)
    s.stdin.close()
    if s.wait():
        raise SystemExit(f"gönderilemedi: {hedef_kok}")


def kur() -> None:
    p = pod()
    o = r.ssh(p, KURULUM, timeout=3600)
    if "KURULDU" not in o.stdout:
        raise SystemExit("kurulum hatası: " + o.stderr[-800:])
    b = r.BURASI
    gonder([(f, f.name) for f in [b / "makine" / "egit.sh", *(b / "egitim").glob("*.py")]], "/workspace/paket_egit")
    gonder([(b / "makine" / "asama23.sh", "asama23.sh")], "/workspace")
    r.ssh(p, "sed -i 's/\\r$//' /workspace/asama23.sh /workspace/paket_egit/egit.sh", timeout=60)
    print("kuruldu", flush=True)


def yukle() -> None:
    p = pod()
    r.veri_gonder(p)                                              # veri/egitim → /workspace/egitim (eksikler)
    v = KOK / "veri" / "egitim_vurgu"
    gonder([(f, f.name) for f in v.glob("*.txt")], "/workspace/egitim_vurgu")
    agirlik = KOK / "cikti" / "ogrenci" / "epoch_2nd_00004.pth"
    gonder([(agirlik, "epoch_2nd_00004.pth")], "/workspace/StyleTTS2/Models/Turkce")
    __import__("plbert").plbert_egitime_gonder()                                     # Utils/PLBERT_tr + UC_HAZIR
    o = r.ssh(p, "ls /workspace/egitim/wavs | wc -l; wc -l /workspace/egitim_vurgu/*.txt; "
                 "ls -la /workspace/StyleTTS2/Models/Turkce /workspace/StyleTTS2/Utils/PLBERT_tr; ls /workspace/UC_HAZIR",
              timeout=120)
    print(o.stdout, flush=True)


def baslat() -> None:
    p = pod()
    r.ssh(p, "pkill -f '[a]sama23.sh'; pkill -f '[t]rain_finetune'; true", timeout=60)
    o = r.ssh(p, f"cd /workspace && (GRACE={r.GRACE_DK} setsid nohup bash asama23.sh < /dev/null > /dev/null 2>&1 &)",
              timeout=120)
    print("3. aşama başladı" + ("" if o.returncode == 0 else f" HATA {o.stderr[-300:]}"), flush=True)


if __name__ == "__main__":
    adim = sys.argv[1] if len(sys.argv) > 1 else "hepsi"
    for ad, f in (("ac", ac), ("kur", kur), ("yukle", yukle), ("baslat", baslat)):
        if adim in (ad, "hepsi"):
            print(f"== {time.strftime('%H:%M')} {ad}", flush=True)
            f()
