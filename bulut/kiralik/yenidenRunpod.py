# -*- coding: utf-8 -*-
"""
KAGGLE YENİDEN ÜRETİMİ RUNPOD'DA (20.10) — Kaggle kotası dolunca (ya da acele edilince) bulut/kaggle/yeniden.py
DEĞİŞMEDEN kiralık 4090'larda koşar: makinede /kaggle/input/seslen-yeniden (girdi kopyası) ve /kaggle/working
(/workspace/cikti) kurulur. Her makine bir plan parçası; bitince 30 dk sonra kendini durdurur (bilgisayar kapalı olabilir).

  bol      veri/plan_yeniden'deki parçaları N'e böler (plan6, plan7, … yeniden yazılır) ve girdi klasörünü günceller
  baslat   N makine açar (rol "uretim"), girdiyi yükler, üretimi başlatır
  durum    makinelerdeki ilerleme
  indir    biten makineleri (duranları GPU'suz açarak) indirir → cikti/kaggle/indirilen/seslen-yeniden-uretim-<parça>/,
           sonra siler

    python bulut/kiralik/yenidenRunpod.py bol --adet 3 && python bulut/kiralik/yenidenRunpod.py baslat
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod as r  # noqa: E402

KOK = r.KOK
PLAN = KOK / "veri" / "plan_yeniden"
GIRDI = KOK / "cikti" / "kaggle" / "yeniden"           # kaggleIs.yeniden'in veri seti klasörü
INDIRILEN = KOK / "cikti" / "kaggle" / "indirilen"
BETIK = r"""#!/bin/bash
cd /workspace; exec >> /workspace/uret.log 2>&1
echo "== $(date) üretim başlıyor"
mkdir -p /kaggle/input /kaggle/tmp /workspace/cikti
cp -r /workspace/girdi /kaggle/input/seslen-yeniden
ln -sfn /workspace/cikti /kaggle/working
# voxcpm torch>=2.5 ister; serbest kurulum en yeni torch'u (CUDA 12.8+ sürücü) getiriyor → sürücüyle uyumlu 2.6 cu124 önce
pip install -q torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # kalıbın torchvision'ı torch 2.4'lük: nms hatası
pip install -q torchcodec==0.2.1 voxcpm==2.0.3 faster-whisper==1.2.1 soxr librosa "transformers==4.51.3"   # 5.x torch 2.6 ile yüklenmiyor
python -c "import torch; assert torch.cuda.is_available(), 'CUDA yok'; from transformers import Wav2Vec2ForCTC; import voxcpm; from faster_whisper import WhisperModel; print('kitaplıklar tamam', torch.__version__, torch.cuda.get_device_name(0))" || { echo "çıkış CUDA" > /workspace/URETIM_BITTI; runpodctl stop pod "$(cat /workspace/.pod_id)"; exit 1; }
# faster-whisper (ctranslate2) cuBLAS / cuDNN'i sistemde arar: pip'in nvidia kitaplıkları yola eklenir
export LD_LIBRARY_PATH=$(python -c "import nvidia, os; b = nvidia.__path__[0]; print(':'.join(os.path.join(b, d, 'lib') for d in os.listdir(b) if os.path.isdir(os.path.join(b, d, 'lib'))))"):$LD_LIBRARY_PATH
for n in $PARCALAR; do echo "== $(date) parça $n"; python -u /workspace/yeniden_calis_$n.py; done
echo "çıkış $?" > /workspace/URETIM_BITTI
echo "== $(date) bitti; 30 dk sonra makine duracak"
sleep 1800; runpodctl stop pod "$(cat /workspace/.pod_id)"
"""


def parcalar() -> list[int]:
    return sorted(int(f.stem.removeprefix("plan")) for f in PLAN.glob("plan*.jsonl"))


def bol(adet: int) -> None:
    nolar = parcalar()
    satir = [x for n in nolar for x in (PLAN / f"plan{n}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    satir.sort(key=lambda x: json.loads(x)["ses"])          # ses ses ardışık; parçalar yaklaşık eşit
    for f in PLAN.glob("plan*.jsonl"):
        f.unlink()
    bas, boy = nolar[0], -(-len(satir) // adet)
    for i in range(adet):
        p = satir[i * boy:(i + 1) * boy]
        (PLAN / f"plan{bas + i}.jsonl").write_text("\n".join(p) + "\n", encoding="utf-8")
        print(f"plan{bas + i}: {len(p)} cümle ({', '.join(sorted({json.loads(x)['ses'] for x in p}))})", flush=True)
    for f in GIRDI.glob("plan*.jsonl"):
        f.unlink()
    for f in PLAN.glob("plan*.jsonl"):
        shutil.copy(f, GIRDI)


def _gonder(pod: dict, kaynak: Path, hedef: str) -> None:
    s = subprocess.Popen(["ssh", *r.ssh_arg(pod), f"root@{pod['ip']}", f"mkdir -p {hedef} && tar xf - -C {hedef}"],
                         stdin=subprocess.PIPE)
    with tarfile.open(fileobj=s.stdin, mode="w|") as t:
        for f in kaynak.rglob("*"):
            if f.is_file():
                t.add(f, arcname=f.relative_to(kaynak).as_posix())
    s.stdin.close()
    if s.wait():
        raise SystemExit(f"{pod['ad']}: girdi gönderilemedi")


def baslat() -> None:

    nolar = parcalar()
    ayar = json.loads((PLAN / "ayar.json").read_text(encoding="utf-8"))
    r.GPULAR[:] = ["NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA GeForce RTX 3090",
                   "NVIDIA RTX A5000", "NVIDIA L4", "NVIDIA RTX 4000 Ada Generation"]       # 20.10: 4090 yoktu
    # 20.10: boş GPU azdı (bir makine açıldı, ikincisi "no instances"): açılabilen kadar makine, parçalar onlara dağıtılır
    r.ac(0, guvenli=False, rol="uretim")              # önceden açılmış ama SSH'ı beklenmemiş makine
    for _ in range(len(nolar) - len([x for x in r.podlar() if x.get("rol") == "uretim"])):
        try:
            r.ac(1, guvenli=False, rol="uretim")         # önce Community (Secure da çoğu zaman dolu)
        except SystemExit as e:
            print(f"makine açılamadı: {str(e)[:120]}", flush=True)
            break
    p = r.podlar()
    for x in [x for x in p if x.get("rol") == "uretim" and not x.get("ip")]:     # SSH vermeyen makine ücret yazmasın
        print(f"{x['ad']}: SSH yok, siliniyor", flush=True)
        r.api(f"/pods/{x['id']}", "DELETE")
        p.remove(x)
    for x in [x for x in p if x.get("rol") == "uretim" and "parca" not in x and x.get("ip")]:
        o = r.ssh(x, "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'", timeout=60).stdout
        surum = float((o.split(":")[-1].strip() or "0").rsplit(".", 0)[0][:4] or 0)
        if surum < 12.4:                              # torch 2.6 cu124 çalışmaz
            print(f"{x['ad']}: sürücü CUDA {surum} eski, siliniyor", flush=True)
            r.api(f"/pods/{x['id']}", "DELETE")
            p.remove(x)
    r.podlar_yaz(p)
    yeni = [x for x in p if x.get("rol") == "uretim" and "parca" not in x and x.get("ip")]
    if not yeni:
        raise SystemExit("hiç makine açılamadı")
    betik = KOK / "cikti" / "uret_yeniden.sh"
    betik.write_text(BETIK, encoding="utf-8", newline="\n")
    for i, pod in enumerate(yeni):
        benim = nolar[i::len(yeni)]
        pod["parca"] = benim[0]
        pod["parcalar"] = benim
        _gonder(pod, GIRDI, "/workspace/girdi")
        s_ = subprocess.Popen(["ssh", *r.ssh_arg(pod), f"root@{pod['ip']}", "tar xf - -C /workspace"], stdin=subprocess.PIPE)
        with tarfile.open(fileobj=s_.stdin, mode="w|") as t:
            for n in benim:
                gecici = KOK / "cikti" / f"yeniden_calis_{n}.py"
                gecici.write_text(f"AYAR = {({**ayar, 'parca': n, 'sure_dk': 1400})!r}\n"
                                  + (KOK / "bulut" / "kaggle" / "yeniden.py").read_text(encoding="utf-8"), encoding="utf-8")
                t.add(gecici, arcname=f"yeniden_calis_{n}.py")
            t.add(betik, arcname="uret_yeniden.sh")
        s_.stdin.close()
        s_.wait()
        o = r.ssh(pod, f"cd /workspace && (PARCALAR='{' '.join(map(str, benim))}' setsid nohup bash uret_yeniden.sh "
                       "< /dev/null > /dev/null 2>&1 &)", timeout=60)
        print(f"{pod['ad']}: parça {benim} başladı" + ("" if o.returncode == 0 else f" HATA {o.stderr[-200:]}"), flush=True)
    r.podlar_yaz(p)


def durum() -> None:
    for pod in r.podlar():
        if pod.get("rol") != "uretim" or "parca" not in pod:
            continue
        o = r.ssh(pod, "cat /workspace/URETIM_BITTI 2>/dev/null; for f in /workspace/cikti/veri/*/olcum.jsonl; do "
                       "echo \"$f $(wc -l < $f)\"; done 2>/dev/null; tail -n 2 /workspace/uret.log", timeout=60)
        print(f"== {pod['ad']} (parça {pod['parca']}):\n{o.stdout.strip() or o.stderr.strip()[-200:]}", flush=True)


def indir() -> None:
    kalan = []
    for pod in r.podlar():
        if pod.get("rol") != "uretim" or "parca" not in pod:
            kalan.append(pod)
            continue
        bilgi = r.api(f"/pods/{pod['id']}")
        if bilgi.get("desiredStatus") != "RUNNING":       # kendini durdurmuş: YALNIZ GPU'suz aç (22.10: GPU'lu açılıp
            o = r._graphql(f'mutation {{ podResume(input: {{podId: "{pod["id"]}", gpuCount: 0}}) {{ id }} }}')   # 11 sa boşta kaldı)
            if o.get("errors"):
                print(f"{pod['ad']}: açılamadı ({o['errors'][0].get('message', '')[:80]})", flush=True)
                kalan.append(pod)
                continue
        for _ in range(60):                               # adres her açılışta değişir: her seferinde yenilenir
            b = r.api(f"/pods/{pod['id']}")
            if b.get("publicIp") and (b.get("portMappings") or {}).get("22"):
                pod.update(ip=b["publicIp"], port=b["portMappings"]["22"])
                if r.ssh(pod, "true", timeout=30).returncode == 0:
                    break
            time.sleep(10)
        if r.ssh(pod, "test -f /workspace/URETIM_BITTI", timeout=60).returncode != 0:
            print(f"{pod['ad']}: henüz bitmedi", flush=True)
            kalan.append(pod)
            continue
        hedef = INDIRILEN / f"seslen-yeniden-uretim-{pod['parca']}"
        if hedef.exists():
            shutil.rmtree(hedef)
        hedef.mkdir(parents=True)
        s = subprocess.Popen(["ssh", *r.ssh_arg(pod), f"root@{pod['ip']}", "cd /workspace/cikti && tar cf - veri ref; "
                              "cat /workspace/uret.log > /dev/null"], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=s.stdout, mode="r|") as t:
            t.extractall(hedef)
        s.wait()
        r.api(f"/pods/{pod['id']}", "DELETE")
        print(f"{pod['ad']}: parça {pod['parca']} indirildi → {hedef}, makine silindi", flush=True)
    r.podlar_yaz(kalan)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("adim", choices=["bol", "baslat", "durum", "indir"])
    ap.add_argument("--adet", type=int, default=3)
    a = ap.parse_args()
    {"bol": lambda: bol(a.adet), "baslat": baslat, "durum": durum, "indir": indir}[a.adim]()
