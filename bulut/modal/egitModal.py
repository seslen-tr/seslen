# -*- coding: utf-8 -*-
"""
MODAL EĞİTİM (15.10) — RunPod yerine Modal: her ay 30 $ ücretsiz kredi, yalnız HTTPS (443'ten başka portu kapalı ağda
da çalışır), saniye başı ücret. bulut/kiralik/asama4_odak.py + odak.sh ile aynı ince ayar: StyleTTS2 train_finetune,
config bulut/kiralik/egitim/ayar.py, cuDNN kapalı (egitim/sitecustomize.py). Liste yenile.py'nin "liste" adımından
(veri/egitim_vurgu_<tur>/: train/val/OOD + wavlar.txt).

Adımlar:
  yukle   listedeki wav'lar ~1 GB'lık tar parçalarıyla "seslen-egitim" hacmine (orada açılır), başlangıç ağırlığı,
          Türkçe PL-BERT, liste dosyaları
  hiz     200 adımlık deneme: adım süresi → bütün turun tahmini süresi ve ücreti
  egit    ayrık başlar (bilgisayar kapansa da sürer); ağırlık ve train.log hacimde
  izle    train.log'un son satırları
  indir   Models/<KAYIT>/epoch_2nd_00000.pth → cikti/ogrenci/<tur>-epoch_2nd_00000.pth (+ train.log, config)

Kurulum (bir kez): modal.com'da hesap, sonra  python -m modal setup  (anahtar ~/.modal.toml'a yazılır).
    python bulut/modal/egitModal.py yukle|hiz|egit|izle|indir [--gpu A100] [--tur t9]
Hız denemesi veri gelmeden: --tur hiz (veri/egitim_vurgu_hiz: t8 listesinden 1000 satır).
"""
from __future__ import annotations

import sys
import tarfile
import time
from pathlib import Path

import modal

_YOL = Path(__file__).resolve()
# Modal kapsayıcısında bu dosya /root/egitModal.py olarak durur (parents[2] yok): yerel yollar yalnız bilgisayarda anlamlı
KOK = _YOL.parents[2] if len(_YOL.parents) > 2 else _YOL.parent
EGITIM = KOK / "bulut" / "kiralik" / "egitim"
sys.path.insert(0, str(KOK / "bulut" / "kiralik"))

TUR = "t9"
KAYIT = "Turkce" + TUR[1:]
BASLANGIC = KOK / "cikti" / "ogrenci" / "t8-epoch_2nd_00000.pth"   # --tur verilince yenile.TURLAR[tur]["baslangic"]
LISTE = KOK / "veri" / f"egitim_vurgu_{TUR}"
PLBERT = Path.home() / "Desktop" / "ses_projeleri" / "turkce-plbert" / "model"
GPU = "A100"        # GPU için Modal hesabında ödeme yöntemi şart (hediye kredi yine de 0 $); --gpu ile değişir
SAAT_UCRET = {"A100": 2.10, "A100-80GB": 2.50, "A10G": 1.10, "L4": 0.80, "H100": 3.95}   # $/sa, yaklaşık
PARCA_GB = 1.0
ST = "/workspace/StyleTTS2"

app = modal.App("seslen-egitim")
hacim = modal.Volume.from_name("seslen-egitim", create_if_missing=True)
imaj = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "libsndfile1")
    .pip_install("torch==2.4.1", "torchaudio==2.4.1", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("munch", "einops", "einops-exts", "pydub", "nltk", "accelerate<1.3", "transformers==4.46.3",
                 "soundfile", "librosa", "pandas", "tensorboard", "matplotlib", "pyyaml", "cython", "huggingface_hub")
    .run_commands(
        "pip install --no-build-isolation git+https://github.com/resemble-ai/monotonic_align.git",
        f"git clone -q --depth 1 https://github.com/yl4579/StyleTTS2 {ST}",
        "python -c \"from huggingface_hub import hf_hub_download as h; "
        f"h('yl4579/StyleTTS2-LJSpeech', 'Models/LJSpeech/config.yml', local_dir='{ST}')\"",
        # odak.sh ile aynı: yeni transformers'ta position_ids anahtarı yok
        f"sed -i 's/    del new_state_dict\\[\"embeddings.position_ids\"\\]/"
        f"    new_state_dict.pop(\"embeddings.position_ids\", None)/' {ST}/Utils/PLBERT/util.py",
    )
    .add_local_dir(str(EGITIM), "/root/paket_egit", ignore=["__pycache__"])
)


def _hazirla(tur: str, kayit: str, bas: str = "") -> dict:
    """Makinede: hacimdeki veriyi StyleTTS2'nin beklediği yollara bağlar, ayar.py ile config yazar; ortamı döndürür."""
    import os
    import subprocess
    Path("/workspace/egitim").mkdir(parents=True, exist_ok=True)
    for kaynak, hedef in (("/vol/wavs", "/workspace/egitim/wavs"), ("/vol/baslangic", f"{ST}/Models/Baslangic"),
                          ("/vol/plbert", f"{ST}/Utils/PLBERT_tr"), (f"/vol/Models/{kayit}", f"{ST}/Models/{kayit}")):
        Path(kaynak).mkdir(parents=True, exist_ok=True)
        if not Path(hedef).exists():
            os.symlink(kaynak, hedef)
    ood = Path(f"/vol/liste/{tur}/OOD_texts.txt")
    satir = ood.read_text(encoding="utf-8").splitlines()
    if satir and "|" not in satir[0]:                                     # odak.sh: OOD satırlarına konuşmacı
        ood.write_text("\n".join(s + "|0" for s in satir) + "\n", encoding="utf-8")
    bas = bas or next(Path("/vol/baslangic").glob("*.pth")).name   # hacimde birden çok başlangıç olabilir
    ortam = {**os.environ, "PYTHONPATH": "/root/paket_egit", "BASLANGIC": f"Models/Baslangic/{bas}",
             "KAYIT": f"Models/{kayit}", "PLBERT": "Utils/PLBERT_tr/", "LISTE": f"/vol/liste/{tur}",
             "EPOCH": "1", "DIFF": "0", "JOINT": "0", "BATCH": "2", "MAXLEN": "300", "LR": "5e-5"}
    # 18.10 (RunPod'da makine yeniden başladı, ara kayıt yoktu): her 1500 adımda ara.pth; varsa oradan sürer
    subprocess.run(["python", "/root/paket_egit/yama.py", f"{ST}/train_finetune.py"], check=True)
    ara = Path(f"/vol/Models/{kayit}/ara.pth")
    if ara.exists():
        import torch
        ortam["DEVAM"] = f"Models/{kayit}/ara.pth"
        ortam["ATLA"] = str(torch.load(ara, map_location="cpu", weights_only=False)["adim"])
        print(f"ara kayıttan sürdürülüyor: adım {ortam['ATLA']}", flush=True)
    subprocess.run(["python", "/root/paket_egit/ayar.py"], env=ortam, check=True, cwd=ST)
    return ortam


def _calistir(ortam: dict, kayit: str, dur_adim: int = 0, butce_sa: float = 0) -> str:
    """train_finetune'u çalıştırır; 5 dakikada bir hacmi kaydeder (izle görebilsin). dur_adim: o adımda durdur.
    butce_sa: ilk başlangıçtan (yeniden denemeler dahil) bu kadar saat sonra durdur — kredi aşılmasın."""
    import re
    import subprocess
    log = Path(f"/vol/Models/{kayit}/train.log")
    ilk = Path(f"/vol/Models/{kayit}/ilk_baslangic")
    if butce_sa and not ilk.exists():
        ilk.write_text(str(time.time()))
    bitis = float(ilk.read_text()) + butce_sa * 3600 if butce_sa else float("inf")
    cikti = open(f"/vol/Models/{kayit}/egit.out", "a")
    p = subprocess.Popen(["python", "train_finetune.py", "--config_path", "Configs/config_tr.yml"], cwd=ST,
                         env=ortam, stdout=cikti, stderr=subprocess.STDOUT)
    son_kayit = time.time()
    while p.poll() is None:
        time.sleep(20)
        if dur_adim and log.exists():
            adim = [int(m) for m in re.findall(r"Step \[(\d+)/", log.read_text(errors="ignore"))]
            if adim and max(adim) >= dur_adim:
                p.terminate()
                p.wait()
                break
        if time.time() > bitis:
            print("süre bütçesi doldu, durduruluyor (ara kayıttan sürdürülebilir)", flush=True)
            p.terminate()
            p.wait()
            break
        if time.time() - son_kayit > 300:
            hacim.commit()
            son_kayit = time.time()
    cikti.close()
    hacim.commit()
    return log.read_text(errors="ignore") if log.exists() else Path(f"/vol/Models/{kayit}/egit.out").read_text()[-3000:]


@app.function(image=imaj, volumes={"/vol": hacim}, timeout=3600, cpu=4)
def tar_ac() -> int:
    """Hacimdeki /vol/tar/*.tar → /vol/wavs; açılan parça silinir."""
    say = 0
    for t in sorted(Path("/vol/tar").glob("*.tar")):
        with tarfile.open(t) as f:
            f.extractall("/vol/wavs")
            say += len(f.getmembers())
        t.unlink()
    hacim.commit()
    return say


@app.function(image=imaj, gpu=GPU, volumes={"/vol": hacim}, timeout=2 * 3600)
def hiz_f(tur: str, kayit: str) -> str:
    import shutil
    ortam = _hazirla(tur, kayit + "_hiz")
    shutil.rmtree(f"/vol/Models/{kayit}_hiz", ignore_errors=True)
    Path(f"/vol/Models/{kayit}_hiz").mkdir(parents=True, exist_ok=True)
    return _calistir(ortam, kayit + "_hiz", dur_adim=200)


BUTCE_SA = 12.0          # A100 ~2,1 $/sa → en çok ~25 $: aylık 30 $ hediye kredi aşılmaz


@app.function(image=imaj, gpu=GPU, volumes={"/vol": hacim}, timeout=int(BUTCE_SA * 3600) + 1800,
              retries=modal.Retries(max_retries=2, initial_delay=60.0))
def egit_f(tur: str, kayit: str, bas: str = "", butce_sa: float = BUTCE_SA) -> str:
    """Çökerse (makine kesintisi) Modal yeniden dener; ara kayıttan sürer, bütçe ilk başlangıçtan sayılır."""
    ortam = _hazirla(tur, kayit, bas)
    return _calistir(ortam, kayit, butce_sa=butce_sa)[-2000:]


def yukle() -> None:
    wavs = KOK / "veri" / "egitim" / "wavs"
    tek = [w for w in (LISTE / "wavlar.txt").read_text(encoding="utf-8").splitlines() if w]
    var = set()
    try:
        var = {e.path.removeprefix("wavs/"): e.size for e in hacim.iterdir("wavs", recursive=True)}
    except Exception:
        pass
    # boyutla: yeni sesin aynı adlı kaydı (t10 Damla ne0151.flac) eskisinin yerine geçsin
    eksik = [w for w in tek if var.get(w) != (wavs / w).stat().st_size]
    print(f"{len(tek)} wav, hacimde {len(tek) - len(eksik)}; gönderilecek {len(eksik)}", flush=True)
    gecici = KOK / "cikti" / "modal_tar"
    gecici.mkdir(parents=True, exist_ok=True)
    parcalar, boyut, i = [], 0, 0
    t = None
    for w in eksik:
        if t is None:
            yol = gecici / f"parca{i:03d}.tar"
            t, boyut = tarfile.open(yol, "w"), 0
            parcalar.append(yol)
        t.add(wavs / w, arcname=w)
        boyut += (wavs / w).stat().st_size
        if boyut > PARCA_GB * 1e9:
            t.close()
            t, i = None, i + 1
    if t:
        t.close()
    t0 = time.time()
    with hacim.batch_upload(force=True) as b:
        for yol in parcalar:
            b.put_file(str(yol), f"tar/{yol.name}")
        for ad in ("train_list.txt", "val_list.txt", "OOD_texts.txt"):
            b.put_file(str(LISTE / ad), f"liste/{TUR}/{ad}")
        mevcut = {}
        try:
            mevcut = {e.path: e.size for d in ("baslangic", "plbert") for e in hacim.listdir(d)}
        except Exception:
            pass
        adim = sorted(PLBERT.glob("step_*.t7"), key=lambda p: int(p.stem.split("_")[1]))[-1]
        for yerel, uzak in ((BASLANGIC, f"baslangic/{BASLANGIC.name}"), (adim, f"plbert/{adim.name}"),
                            (PLBERT / "config.yml", "plbert/config.yml")):
            if mevcut.get(uzak) != yerel.stat().st_size:      # 750 MB'lık ağırlık bir kez gider
                b.put_file(str(yerel), uzak)
    print(f"yüklendi ({(time.time() - t0) / 60:.1f} dk); hacimde açılıyor", flush=True)
    with modal.enable_output(), app.run():
        print(f"açılan dosya: {tar_ac.remote()}", flush=True)
    for yol in parcalar:
        yol.unlink()


def hiz(gpu: str) -> None:
    import re
    with modal.enable_output(), app.run():
        log = hiz_f.with_options(gpu=gpu).remote(TUR, KAYIT)
    zaman = [(time.mktime(time.strptime(m[0], "%Y-%m-%d %H:%M:%S")), int(m[1]), int(m[2]))
             for m in re.findall(r"INFO:(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+: Epoch \[\d+/\d+\], Step \[(\d+)/(\d+)\]", log)]
    if len(zaman) < 2:
        print("adım süresi ölçülemedi; son çıktı:\n" + log[-2000:])
        return
    sn_adim = (zaman[-1][0] - zaman[0][0]) / (zaman[-1][1] - zaman[0][1])
    toplam = zaman[-1][2]
    sa = sn_adim * toplam / 3600
    print(f"{gpu}: {sn_adim:.2f} sn/adım × {toplam} adım ≈ {sa:.1f} sa ≈ {sa * SAAT_UCRET.get(gpu, 0):.0f} $", flush=True)


def egit(gpu: str) -> None:
    with modal.enable_output(), app.run(detach=True):
        c = egit_f.with_options(gpu=gpu).spawn(TUR, KAYIT, BASLANGIC.name, BUTCE)
    print(f"eğitim başladı ({gpu}, ayrık): {c.object_id}. İzleme: python bulut/modal/egitModal.py izle", flush=True)


def izle() -> None:
    for ad in ("train.log", "egit.out"):
        try:
            veri = b"".join(hacim.read_file(f"Models/{KAYIT}/{ad}")).decode(errors="ignore")
            print(f"== {ad}\n" + "\n".join(veri.splitlines()[-8:]))
        except Exception as e:
            print(f"== {ad}: yok ({type(e).__name__})")


def indir() -> None:
    hedef = KOK / "cikti" / "ogrenci"
    for uzak, yerel in ((f"Models/{KAYIT}/epoch_2nd_00000.pth", f"{TUR}-epoch_2nd_00000.pth"),
                        (f"Models/{KAYIT}/train.log", f"{TUR}-train.log"),
                        (f"Models/{KAYIT}/config_tr.yml", f"{TUR}-config_tr.yml")):
        try:
            with open(hedef / yerel, "wb") as f:
                for parca in hacim.read_file(uzak):
                    f.write(parca)
            print(f"indirildi: {hedef / yerel}", flush=True)
        except Exception as e:
            print(f"{uzak}: {type(e).__name__}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("adim", choices=["yukle", "hiz", "egit", "izle", "indir"])
    ap.add_argument("--gpu", default=GPU)
    ap.add_argument("--butce", type=float, default=BUTCE_SA, help="egit: en çok saat (kalan krediye göre)")
    ap.add_argument("--tur", default=TUR, help="liste: veri/egitim_vurgu_<tur>; 'hiz' = t8'den 1000 satırlık deneme")
    a = ap.parse_args()
    TUR, KAYIT, LISTE = a.tur, "Turkce" + a.tur.removeprefix("t"), KOK / "veri" / f"egitim_vurgu_{a.tur}"
    BUTCE = a.butce
    import yenile
    if TUR in yenile.TURLAR:                          # t10: başlangıç t9 (t8 değil)
        BASLANGIC = KOK / "cikti" / "ogrenci" / yenile.TURLAR[TUR]["baslangic"]
    {"yukle": yukle, "hiz": lambda: hiz(a.gpu), "egit": lambda: egit(a.gpu), "izle": izle, "indir": indir}[a.adim]()
