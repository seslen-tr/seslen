# -*- coding: utf-8 -*-
"""
RUNPOD — kiralık GPU'da öğretmen verisi üretimi, buradan yönetilir.

Anahtar: C:\\Users\\<sen>\\.runpod\\api_key (depoda / sohbette DEĞİL).
SSH anahtarı: ~/.ssh/sesatolye_runpod (ilk "ac"ta üretilir, makineye PUBLIC_KEY ile gider).

    python bulut/kiralik/runpod.py paketle                 plan + kimlikler + kod → cikti/kiralik/paket.tar.gz
    python bulut/kiralik/runpod.py ac --adet 1             makine(ler) aç (4090), SSH hazır olunca döner
    python bulut/kiralik/runpod.py baslat --sesler ayse,koca [--sinir 6]   (boş = bütün plan, makinelere dağıtılır)
    python bulut/kiralik/runpod.py durum                   ilerleme, hız, harcama
    python bulut/kiralik/runpod.py indir                   çıktıyı cikti/veri/<ses>/ ile birleştirir
    python bulut/kiralik/runpod.py kapat                   makineleri SİLER (disk dahil) — önce indir!
    python bulut/kiralik/runpod.py izle                    bitenleri indirip kapatır (arka planda koşar)

Para yanmasın diye üç kat: (1) izle bitince indirip siler, (2) makine kendi
işi bitince GRACE dk sonra kendini durdurur (calistir.sh), (3) durum harcamayı yazar.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

KOK = Path(__file__).resolve().parents[2]
BURASI = Path(__file__).resolve().parent

# Bu süreç çalıştıkça Windows uykuya geçmesin: 04.10'da bilgisayar 6 saat uyudu, yükleme koptu,
# makine boşta ücret yazdı. (Elle "Uyku" seçilirse yine uyur.)
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
IS = KOK / "cikti" / "kiralik"
PODLAR = IS / "podlar.json"
API = "https://rest.runpod.io/v1"
SSH_ANAHTAR = Path.home() / ".ssh" / "sesatolye_runpod"
GORUNTU = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# Üretim planı klasörü: SESATOLYE_PLAN=plan_yeni (03.10: üzgün / kızgın / fısıltı) ya da plan_deneme
PLAN = Path(__file__).resolve().parents[2] / "veri" / os.environ.get("SESATOLYE_PLAN", "plan")
# sırası öncelik: fiyat/hız; hepsi bf16 (sm80+) ve ≥ 24 GB
GPULAR = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090", "NVIDIA L40S", "NVIDIA GeForce RTX 3090",
          "NVIDIA RTX A6000", "NVIDIA A40"]
GRACE_DK = 30       # bitince bu kadar dk bekleyip durur (yerel izleyici yoksa boşta para yanmasın; izle 10 dk arayla bakıyor)
sys.path.insert(0, str(KOK / "ogretmen"))


def anahtar() -> str:
    return (Path.home() / ".runpod" / "api_key").read_text(encoding="utf-8").strip()


def api(yol: str, yontem: str = "GET", govde: dict | None = None):
    istek = urllib.request.Request(f"{API}{yol}", method=yontem,
                                   data=json.dumps(govde).encode() if govde is not None else None,
                                   headers={"Authorization": f"Bearer {anahtar()}", "Content-Type": "application/json",
                                            # Cloudflare urllib varsayılan kimliğini engelliyor (hata 1010)
                                            "User-Agent": "sesatolye/1.0"})
    try:
        with urllib.request.urlopen(istek, timeout=60) as r:
            veri = r.read()
            return json.loads(veri) if veri else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"RunPod {yontem} {yol}: {e.code} {e.read().decode(errors='replace')[:500]}")


def podlar() -> list[dict]:
    return json.loads(PODLAR.read_text(encoding="utf-8")) if PODLAR.exists() else []


def podlar_yaz(p: list[dict]) -> None:
    IS.mkdir(parents=True, exist_ok=True)
    PODLAR.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")


def ssh_arg(pod: dict, port: str = "-p") -> list[str]:
    """port: ssh "-p", scp "-P" ister."""
    bilinen = IS / "known_hosts"
    return ["-i", str(SSH_ANAHTAR), port, str(pod["port"]), "-o", "StrictHostKeyChecking=no",
            "-o", f"UserKnownHostsFile={bilinen}", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]


def ssh(pod: dict, komut: str, yakala: bool = True, **k) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *ssh_arg(pod), f"root@{pod['ip']}", komut], capture_output=yakala,
                          text=True, encoding="utf-8", errors="replace", **k)


def paketle() -> None:
    import kadro
    d = IS / "paket"
    if d.exists():
        shutil.rmtree(d)
    (d / "ref").mkdir(parents=True)
    shutil.copytree(PLAN, d / "plan")
    for ses, aday in kadro.secilenler().items():
        shutil.copy(KOK / "cikti" / "tazele" / aday / "ref2.wav", d / "ref" / f"{ses}.wav")
    for f in (KOK / "araclar" / "olcu" / "olcu.py", KOK / "ogretmen" / "duygu.py", KOK / "ogretmen" / "kadro.py",
              BURASI / "uret.py", BURASI / "makine" / "calistir.sh"):
        shutil.copy(f, d)
    with tarfile.open(IS / "paket.tar.gz", "w:gz") as t:
        t.add(d, arcname="paket")
    print(f"paket hazır: {IS / 'paket.tar.gz'} ({(IS / 'paket.tar.gz').stat().st_size / 1e6:.1f} MB), "
          f"{len(list((d / 'plan').glob('*.jsonl')))} ses")


def ac(adet: int, guvenli: bool, rol: str = "") -> None:
    if not SSH_ANAHTAR.exists():
        SSH_ANAHTAR.parent.mkdir(exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "sesatolye", "-f", str(SSH_ANAHTAR)], check=True)
    acik = SSH_ANAHTAR.with_suffix(".pub").read_text(encoding="utf-8").strip()
    p = podlar()
    for i in range(adet):
        govde = {"name": f"sesatolye-{len(p) + 1}", "imageName": GORUNTU, "gpuTypeIds": GPULAR, "gpuCount": 1,
                 "cloudType": "SECURE" if guvenli else "COMMUNITY", "containerDiskInGb": 40, "volumeInGb": 60,
                 "volumeMountPath": "/workspace", "ports": ["22/tcp"], "env": {"PUBLIC_KEY": acik}}
        # 20.10: allowedCudaVersions süzgeciyle hiç makine çıkmıyordu; sürücü SSH açılınca denetlenir (en az CUDA 12.4)
        if rol == "egitim":     # 18.10: makine gece yeniden başladı, eğitim öldü → açılışta devam.sh sürdürür
            govde["dockerStartCmd"] = ["bash", "-c", "(sleep 45; [ -f /workspace/devam.sh ] && bash /workspace/devam.sh) "
                                       ">> /workspace/devam.log 2>&1 & exec /start.sh"]
        # önce Community (ucuz); yer yoksa Secure. İş kaldığı yerden sürdüğü için Community yeterli.
        try:
            r = api("/pods", "POST", govde)
        except SystemExit as e:
            if govde["cloudType"] == "SECURE":
                raise
            print(f"Community'de yer yok ({str(e)[:120]}); Secure deneniyor", flush=True)
            r = api("/pods", "POST", {**govde, "cloudType": "SECURE"})
        print(f"açıldı: {r['id']} ({r.get('costPerHr', '?')} $/sa)", flush=True)
        p.append({"id": r["id"], "ad": govde["name"], "sesler": [], "acilis": time.time(),
                  "saatlik": r.get("adjustedCostPerHr") or r.get("costPerHr"), **({"rol": rol} if rol else {})})
        podlar_yaz(p)
    for pod in p:
        if pod.get("ip"):
            continue
        for _ in range(60):
            r = api(f"/pods/{pod['id']}")
            if r.get("publicIp") and (r.get("portMappings") or {}).get("22"):
                pod.update(ip=r["publicIp"], port=r["portMappings"]["22"])
                if ssh(pod, "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader").returncode == 0:
                    break
                pod.pop("ip")
            time.sleep(10)
        else:
            raise SystemExit(f"{pod['id']}: SSH açılmadı (10 dk)")
        # Kendini durdurma: SSH'tan başlatılan işler kapsayıcının ortamını (RUNPOD_POD_ID, anahtar)
        # GÖRMÜYOR — 30.09'da bu yüzden iki makine saatlerce boşta kaldı. Anahtar runpodctl
        # ayarına, kimlik dosyaya yazılır ve hemen sınanır.
        o = ssh(pod, f"runpodctl config --apiKey {anahtar()} > /dev/null 2>&1; echo {pod['id']} > /workspace/.pod_id; "
                     f"runpodctl get pod {pod['id']} 2>&1 | tail -1")
        if pod["id"] not in o.stdout:
            print(f"UYARI: {pod['ad']} kendini durduramayabilir (runpodctl: {o.stdout.strip()[:120]}) — izle açık kalsın")
        surum = ssh(pod, "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'").stdout.split(":")[-1].strip()
        if surum and tuple(map(int, surum.split("."))) < (12, 4):      # kalıptaki torch cu124
            api(f"/pods/{pod['id']}", "DELETE")
            podlar_yaz([x for x in podlar() if x["id"] != pod["id"]])
            raise SystemExit(f"{pod['ad']}: sürücü CUDA {surum} eski (en az 12.4), makine silindi")
        print(f"{pod['ad']} hazır: {ssh(pod, 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader').stdout.strip()}")
        podlar_yaz(p)


def baslat(sesler: list[str], sinir: int, aday: int, isci: int, pod_id: str | None = None) -> None:
    p = [x for x in podlar() if x.get("ip") and x.get("rol") == "uretim"]
    if not p:
        raise SystemExit("açık üretim makinesi yok: önce 'ac'")
    if pod_id:      # tek makineyi AYNI gruplarla yeniden başlat (kaldığı yerden sürer)
        p = [x for x in p if x["id"] == pod_id]
        yuva = {(p[0]["id"], i): g.split(",") for i, g in enumerate(p[0]["gruplar"])}
        isci, sesler = len(yuva), []
    elif not sesler:
        sesler = [f.stem for f in PLAN.glob("*.jsonl")]
    # makine × işçi yuvalarına cümle sayısına göre dengeli dağıt (büyükten küçüğe, en boş yuvaya)
    if not pod_id:
        yuva = {(x["id"], i): [] for x in p for i in range(isci)}
    yuk = {k: 0 for k in yuva}
    boy = {s: len((PLAN / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()) for s in sesler}
    for s in sorted(sesler, key=boy.get, reverse=True):
        hedef = min(yuk, key=yuk.get)
        yuva[hedef].append(s)
        yuk[hedef] += min(boy[s], sinir) if sinir else boy[s]
    for pod in p:
        gruplar = [",".join(yuva[(pod["id"], i)]) for i in range(isci) if yuva[(pod["id"], i)]]
        if not gruplar:
            continue
        # scp yerine ssh + cat: Windows'un scp'si (SFTP kipi) bu makinelerde takılıyordu
        with open(IS / "paket.tar.gz", "rb") as f:
            subprocess.run(["ssh", *ssh_arg(pod), f"root@{pod['ip']}", "cat > /workspace/paket.tar.gz"],
                           stdin=f, check=True, capture_output=True, timeout=600)
        ek = f"--aday {aday}" + (f" --sinir {sinir}" if sinir else "")
        # eski işi ayrı çağrıda durdur: aynı komut satırında "calistir.sh" geçince pkill kendi kabuğunu öldürüyordu
        ssh(pod, "pkill -f '[c]alistir.sh'; pkill -f '[p]ip install'; pkill -f '[u]ret.py'; true", timeout=60)
        komut = (f"cd /workspace && rm -rf paket && tar xzf paket.tar.gz && rm -f cikti/BITTI && "
                 f"(SESLER='{';'.join(gruplar)}' EK='{ek}' GRACE={GRACE_DK} "
                 f"setsid nohup bash paket/calistir.sh < /dev/null > /dev/null 2>&1 &)")
        # parantez: yalnız iş arka plana düşsün; && zincirinin tamamı düşerse ssh kanalı açık kalıp bekliyordu
        r = ssh(pod, komut, timeout=120)
        pod["sesler"] = [s for g in gruplar for s in g.split(",")]
        pod["gruplar"] = gruplar
        print(f"{pod['ad']}: başladı → {' | '.join(gruplar)}"
              + ("" if r.returncode == 0 else f" HATA: {r.stderr[-300:]}"))
    podlar_yaz(podlar_birlestir(p))


def podlar_birlestir(guncel: list[dict]) -> list[dict]:
    g = {x["id"]: x for x in guncel}
    return [g.get(x["id"], x) for x in podlar()]


def durum() -> list[dict]:
    sonuc = []
    toplam_para = 0.0
    for pod in [x for x in podlar() if x.get("rol") == "uretim"]:
        r = api(f"/pods/{pod['id']}")
        saatlik = r.get("adjustedCostPerHr") or r.get("costPerHr") or pod.get("saatlik") or 0
        sure_sa = (time.time() - pod["acilis"]) / 3600
        para = saatlik * sure_sa
        toplam_para += para
        isciler, bitti, son = [], False, ""
        if pod.get("ip") and r.get("desiredStatus") == "RUNNING":
            o = ssh(pod, "for f in /workspace/cikti/ilerleme-*.json; do cat $f 2>/dev/null; echo; done; "
                         "test -f /workspace/cikti/BITTI && echo BITTI; tail -n 1 /workspace/calistir.log 2>/dev/null",
                    timeout=60)
            satir = [s for s in o.stdout.splitlines() if s.strip()]
            isciler = [json.loads(s) for s in satir if s.startswith("{")]
            bitti = "BITTI" in satir
            son = " | ".join(s for s in satir if not s.startswith("{") and s != "BITTI")[-200:]
        biten = sum(i["biten"] for i in isciler)
        ses_sa = sum(i["ses_sn"] for i in isciler) / 3600
        kalan = max(((i["toplam"] - i["biten"]) * i["is_sn"] / max(i["biten"], 1) / 3600 for i in isciler
                     if i.get("biten")), default=0)
        gecen = sum(i["gecen"] for i in isciler)
        toplam = sum(len((PLAN / f"{s}.jsonl").read_text(encoding="utf-8").splitlines())
                     for s in pod.get("sesler", []) if (PLAN / f"{s}.jsonl").exists())
        print(f"{pod['ad']} [{r.get('desiredStatus')}] {len(isciler)} işçi: {biten}/{toplam} cümle "
              f"(geçen {gecen}, ses {ses_sa:.2f} sa)" + (f" · ~{kalan:.1f} sa kaldı" if kalan else "")
              + f" · açık {sure_sa:.1f} sa · {para:.2f} $" + (" · BİTTİ" if bitti else ""))
        if son:
            print(f"    {son}")
        sonuc.append({**pod, "bitti": bitti, "durum": r.get("desiredStatus")})
    print(f"TOPLAM harcama (tahmini): {toplam_para:.2f} $")
    return sonuc


def indir(pod_id: str | None = None) -> None:
    sys.path.insert(0, str(KOK / "araclar" / "olcu"))
    for pod in podlar():
        if (pod_id and pod["id"] != pod_id) or not pod.get("ip"):
            continue
        hedef = IS / "indirilen" / pod["ad"]
        hedef.mkdir(parents=True, exist_ok=True)
        s = subprocess.Popen(["ssh", *ssh_arg(pod), f"root@{pod['ip']}", "cd /workspace/cikti && tar cf - ."],
                             stdout=subprocess.PIPE)
        with tarfile.open(fileobj=s.stdout, mode="r|") as t:
            t.extractall(hedef)
        s.wait()
        for d in sorted(p for p in hedef.iterdir() if (p / "olcum.jsonl").exists()):
            birlestir(d.name, d)
        print(f"{pod['ad']}: indirildi → {hedef}")


def birlestir(ses: str, kaynak: Path) -> None:
    """Kaynak olcum.jsonl + wavs → cikti/veri/<ses>/ (flac olarak; zaten olan kimlik atlanır)."""
    sys.path.insert(0, str(KOK / "ogretmen"))
    from veriUret import metadata_yaz
    yer = KOK / "cikti" / "veri" / ses
    (yer / "wavs").mkdir(parents=True, exist_ok=True)
    var = set()
    if (yer / "olcum.jsonl").exists():
        var = {json.loads(x)["kimlik"] for x in (yer / "olcum.jsonl").read_text(encoding="utf-8").splitlines()}
    eklenen = 0
    with open(yer / "olcum.jsonl", "a", encoding="utf-8") as f:
        for x in (kaynak / "olcum.jsonl").read_text(encoding="utf-8").splitlines():
            o = json.loads(x)
            if o["kimlik"] in var:
                continue
            shutil.copy(kaynak / "wavs" / f"{o['kimlik']}.flac", yer / "wavs" / f"{o['kimlik']}.flac")
            o["yer"] = "runpod"
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            eklenen += 1
    print(f"  {ses}: {eklenen} kayıt eklendi, metadata {metadata_yaz(ses)}")


def kapat(pod_id: str | None = None) -> None:
    """pod_id verilmezse yalnız ÜRETİM makineleri silinir (eğitim / PL-BERT diskleri korunur)."""
    kalan = []
    for pod in podlar():
        if (pod_id and pod["id"] != pod_id) or (not pod_id and pod.get("rol") != "uretim"):
            kalan.append(pod)
            continue
        api(f"/pods/{pod['id']}", "DELETE")
        print(f"{pod['ad']} silindi")
    podlar_yaz(kalan)


def izle(aralik_dk: int = 10) -> None:
    while any(x.get("rol") == "uretim" for x in podlar()):
        for pod in durum():
            if pod["bitti"]:
                indir(pod["id"])
                kapat(pod["id"])
            elif pod["durum"] in ("EXITED", "TERMINATED"):
                print(f"{pod['ad']} durmuş ({pod['durum']}): elle bakılmalı (runpod.py durum)")
                return
        time.sleep(aralik_dk * 60)
    print("bütün makineler bitti ve kapandı")


# ── ÖĞRENCİ EĞİTİMİ (StyleTTS2) ──────────────────────────────────────────────
# Veri: veri/egitim/ (ogrenci/veri/). Kod: bulut/kiralik/makine/egit.sh + egitim/.

def veri_gonder(pod: dict, akis: int = 4) -> None:
    """veri/egitim → /workspace/egitim. Yalnız eksik / boyutu farklı dosyalar; `akis` paralel
    ssh + tar akışı (tek bağlantı bazı veri merkezlerinde ~0,4 MB/sn'de kalıyordu, 01.10)."""
    from concurrent.futures import ThreadPoolExecutor
    kok = KOK / "veri" / "egitim"
    o = ssh(pod, "mkdir -p /workspace/egitim && cd /workspace/egitim && find . -type f -printf '%s %P\\n'", timeout=300)
    if o.returncode:        # sessizce boş liste dönüp her şeyi yeniden yüklüyordu (01.10)
        raise SystemExit(f"uzak dosya listesi alınamadı: {o.stderr[-300:]}")
    uzak = {}
    for s in o.stdout.splitlines():
        boy, _, yol = s.partition(" ")
        uzak[yol] = int(boy)
    eksik = [p for p in sorted(kok.rglob("*")) if p.is_file()
             and uzak.get(str(p.relative_to(kok)).replace("\\", "/")) != p.stat().st_size]
    boyut = sum(p.stat().st_size for p in eksik)
    print(f"yüklenecek: {len(eksik)} dosya, {boyut / 1e9:.2f} GB ({len(uzak)} dosya zaten orada)", flush=True)
    if not eksik:
        return
    t0 = time.time()

    def gonder(parca: list[Path]) -> int:
        s = subprocess.Popen(["ssh", *ssh_arg(pod), f"root@{pod['ip']}", "tar xf - -C /workspace/egitim"],
                             stdin=subprocess.PIPE)
        with tarfile.open(fileobj=s.stdin, mode="w|") as t:
            for p in parca:
                t.add(p, arcname=str(p.relative_to(kok)).replace("\\", "/"))
        s.stdin.close()
        return s.wait()

    with ThreadPoolExecutor(akis) as ex:
        sonuc = list(ex.map(gonder, [eksik[i::akis] for i in range(akis)]))
    if any(sonuc):
        raise SystemExit(f"veri yüklenemedi (akış çıkışları {sonuc}); yeniden çalıştırınca kaldığı yerden sürer")
    dk = (time.time() - t0) / 60
    print(f"veri yüklendi ({dk:.1f} dk, {boyut / 1e6 / max(dk * 60, 1):.1f} MB/sn)", flush=True)


def _egitim_podu() -> dict:
    """Öğrenci eğitimi makinesi: rolü olmayan (ya da "egitim") makine. Yanında PL-BERT (plbert, plbert-eski)
    ve üretim (uretim) makineleri olabiliyor."""
    return next(x for x in podlar() if x.get("ip") and x.get("rol") in (None, "egitim"))


def egit_baslat(epoch: int, batch: int, maxlen: int, devam: str, veri_yukle: bool) -> None:
    pod = _egitim_podu()
    if veri_yukle:
        veri_gonder(pod)
    s = subprocess.Popen(["ssh", *ssh_arg(pod), f"root@{pod['ip']}",
                          "rm -rf /workspace/paket_egit && mkdir -p /workspace/paket_egit && tar xf - -C /workspace/paket_egit"],
                         stdin=subprocess.PIPE)
    with tarfile.open(fileobj=s.stdin, mode="w|") as t:
        for p in [BURASI / "makine" / "egit.sh", *(BURASI / "egitim").glob("*.py")]:
            t.add(p, arcname=p.name)
    s.stdin.close()
    s.wait()
    ssh(pod, "pkill -f '[e]git.sh'; pkill -f '[t]rain_finetune'; true", timeout=60)
    r = ssh(pod, f"cd /workspace && (EPOCH={epoch} BATCH={batch} MAXLEN={maxlen} DEVAM='{devam}' GRACE={GRACE_DK} "
                 f"setsid nohup bash paket_egit/egit.sh < /dev/null > /dev/null 2>&1 &)", timeout=120)
    print(f"{pod['ad']}: eğitim başladı (tur {epoch}, batch {batch}, max_len {maxlen}, devam {devam or '-'})"
          + ("" if r.returncode == 0 else f" HATA {r.stderr[-300:]}"))


def egit_durum() -> None:
    for pod in podlar():
        r = api(f"/pods/{pod['id']}")
        para = (r.get("adjustedCostPerHr") or r.get("costPerHr") or 0) * (time.time() - pod["acilis"]) / 3600
        print(f"{pod['ad']} [{r.get('desiredStatus')}] açık {(time.time() - pod['acilis']) / 3600:.1f} sa · {para:.2f} $")
        if pod.get("ip") and r.get("desiredStatus") == "RUNNING":
            o = ssh(pod, "tail -n 4 /workspace/egit.log; grep -h -E 'Epoch|Validation' /workspace/StyleTTS2/Models/Turkce/train.log "
                         "2>/dev/null | tail -n 3; ls /workspace/StyleTTS2/Models/Turkce/*.pth 2>/dev/null; "
                         "test -f /workspace/EGITIM_BITTI && echo EGITIM_BITTI; "
                         "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", timeout=60)
            print("    " + "\n    ".join(s[-220:] for s in o.stdout.splitlines() if s.strip()))


def egit_indir(dosya: str) -> None:
    """Models/Turkce/<dosya> (ya da "Turkce3/<dosya>", boşsa en son .pth) → cikti/ogrenci/.
    3. aşamanınkiler (Models/Turkce3) yerelde "t3-" önekli."""
    pod = _egitim_podu()
    if not dosya:
        dosya = ssh(pod, "ls -t /workspace/StyleTTS2/Models/Turkce/*.pth | head -1", timeout=60).stdout.strip().rsplit("/", 1)[-1]
    klasor, ad = dosya.split("/", 1) if "/" in dosya else ("Turkce", dosya)
    onek = "" if klasor == "Turkce" else klasor.lower().replace("turkce", "t") + "-"
    hedef = KOK / "cikti" / "ogrenci"
    hedef.mkdir(parents=True, exist_ok=True)
    # yalnız ağırlıklar (optimizer yarısını atıp boyutu küçült)
    komut = (f"cd /workspace/StyleTTS2 && python -c \"import torch; s=torch.load('Models/{klasor}/{ad}', map_location='cpu', "
             f"weights_only=False); torch.save({{'net': s['net'], 'epoch': s['epoch'], 'val_loss': s['val_loss']}}, '/tmp/{ad}')\" "
             f"&& cat /tmp/{ad}")
    with open(hedef / f"{onek}{ad}", "wb") as f:
        subprocess.run(["ssh", *ssh_arg(pod), f"root@{pod['ip']}", komut], stdout=f, check=True)
    for ek in ("config_tr.yml", "train.log"):
        o = ssh(pod, f"cat /workspace/StyleTTS2/Models/{klasor}/{ek} 2>/dev/null || cat /workspace/StyleTTS2/Configs/config_tr.yml",
                timeout=60)
        (hedef / f"{onek}{ek}").write_text(o.stdout, encoding="utf-8")
    print(f"indirildi: {hedef / (onek + ad)} ({(hedef / (onek + ad)).stat().st_size / 1e6:.0f} MB)")


def egit_izle(aralik_dk: int = 5) -> None:
    """Her yeni tur ağırlığını (yalnız 'net') indirir; eğitim bitince sonuncuyu indirip makineyi siler."""
    hedef = KOK / "cikti" / "ogrenci"
    while True:
        try:
            if not (pl := podlar()):
                return
            pod = _egitim_podu()
            durum = api(f"/pods/{pod['id']}").get("desiredStatus")
            if durum != "RUNNING":
                # 06.10: eğitim bilgisayar kapalı/uykudayken bitip makine kendini durduracaktı → uyanınca kurtar
                print(f"{time.strftime('%H:%M')} makine {durum}: kurtarma başlıyor", flush=True)
                egit_kurtar()
                return
            satir = _izle_tur(pod, hedef)
        except (subprocess.TimeoutExpired, OSError, StopIteration, KeyError) as h:
            # 01.10 07:55: tek bir ssh zaman aşımı izleyiciyi öldürdü; 13 saat model inmedi.
            print(f"{time.strftime('%H:%M')} geçici hata: {type(h).__name__} — sonraki turda yeniden", flush=True)
            time.sleep(aralik_dk * 60)
            continue
        print(f"{time.strftime('%H:%M')} {satir[-1] if satir else ''}", flush=True)
        if "HATALI_BITTI" in satir:
            # 01.10: hatalı bitişte de "bitti" sanıp makineyi sildi. Artık silmez; kayıt egit-son.log'da.
            print("eğitim HATAYLA bitti — makine silinmedi (30 dk sonra kendini durdurur); egit-son.log", flush=True)
            return
        if "BITTI" in satir:
            kapat(pod["id"])
            print("eğitim bitti, makine silindi", flush=True)
            return
        time.sleep(aralik_dk * 60)


def _izle_tur(pod: dict, hedef: Path) -> list[str]:
    o = ssh(pod, "ls /workspace/StyleTTS2/Models/Turkce*/*.pth 2>/dev/null; "
             "test -f /workspace/EGITIM_BITTI && ! pgrep -f '[t]rain_finetune' > /dev/null && "
             "{ grep -q 'çıkış 0)' /workspace/EGITIM_BITTI && echo BITTI || echo HATALI_BITTI; }; "
             "grep -o 'Epoch \\[[0-9]*/[0-9]*\\], Step \\[[0-9]*/[0-9]*' /workspace/egit.log | tail -1", timeout=60)
    satir = o.stdout.splitlines()
    # Kaydın sonunu her seferinde yerele kopyala: 01.10'da eğitim çöktü, makine kendini
    # durdurdu ve yeniden açılamadı (sunucuda boş GPU yok) → hata kaydı okunamadı.
    son = ssh(pod, "grep -v -i warn /workspace/egit.log | tail -n 300", timeout=60).stdout
    (IS / "egit-son.log").write_text(son, encoding="utf-8")
    for yol in (s for s in satir if s.endswith(".pth")):
        klasor, ad = yol.rsplit("/", 2)[-2:]
        onek = "" if klasor == "Turkce" else klasor.lower().replace("turkce", "t") + "-"
        if not (hedef / f"{onek}{ad}").exists():
            egit_indir(f"{klasor}/{ad}")
    return satir


def _graphql(sorgu: str) -> dict:
    istek = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": sorgu}).encode(),
                                   headers={"Authorization": f"Bearer {anahtar()}", "Content-Type": "application/json",
                                            "User-Agent": "sesatolye/1.0"})
    with urllib.request.urlopen(istek, timeout=60) as r:
        return json.loads(r.read())


def egit_kurtar(aralik_dk: int = 10) -> None:
    """Eğitim bitip kendini durdurmuş öğrenci makinesini açar, eksik ağırlıkları indirir; başarıyla bittiyse
    siler. Sunucuda boş GPU yoksa önce GPU'suz (yalnız diske erişim) açmayı dener, olmazsa bekleyip yeniden
    dener. 06.10 için yazıldı: eğitim bilgisayar kapalıyken bitecekti."""
    hedef = KOK / "cikti" / "ogrenci"
    pod = _egitim_podu() if any(x.get("rol") == "egitim" for x in podlar()) else None
    if pod is None:
        raise SystemExit("podlar.json'da eğitim makinesi yok")
    while True:
        d = api(f"/pods/{pod['id']}")
        if d.get("desiredStatus") != "RUNNING":
            try:
                api(f"/pods/{pod['id']}/start", "POST")
                print(f"{time.strftime('%H:%M')} GPU'yla açılıyor", flush=True)
            except SystemExit as e:
                print(f"{time.strftime('%H:%M')} GPU'yla açılamadı: {str(e)[-90:]}", flush=True)
                try:
                    o = _graphql(f'mutation {{ podResume(input: {{podId: "{pod["id"]}", gpuCount: 0}}) '
                                 f'{{ id desiredStatus }} }}')
                    if o.get("errors"):
                        raise RuntimeError(o["errors"][0].get("message", "")[:120])
                    print(f"{time.strftime('%H:%M')} GPU'suz açılıyor (yalnız indirme için)", flush=True)
                except (RuntimeError, OSError) as h:
                    print(f"{time.strftime('%H:%M')} GPU'suz da açılamadı: {h}; {aralik_dk} dk sonra yeniden",
                          flush=True)
                    time.sleep(aralik_dk * 60)
                    continue
        for _ in range(60):
            d = api(f"/pods/{pod['id']}")
            if d.get("publicIp") and (d.get("portMappings") or {}).get("22"):
                break
            time.sleep(10)
        else:
            print(f"{time.strftime('%H:%M')} ssh adresi gelmedi; {aralik_dk} dk sonra yeniden", flush=True)
            time.sleep(aralik_dk * 60)
            continue
        pod.update(ip=d["publicIp"], port=d["portMappings"]["22"])
        pl = podlar()
        for x in pl:
            if x["id"] == pod["id"]:
                x.update(ip=pod["ip"], port=pod["port"])
        podlar_yaz(pl)
        time.sleep(20)                                   # sshd ayağa kalksın
        try:
            satir = _izle_tur(pod, hedef)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as h:
            print(f"{time.strftime('%H:%M')} indirme hatası: {type(h).__name__}; yeniden", flush=True)
            time.sleep(60)
            continue
        print(f"{time.strftime('%H:%M')} " + " · ".join(s.rsplit("/", 1)[-1] for s in satir), flush=True)
        if "BITTI" in satir:
            kapat(pod["id"])
            print("ağırlıklar indi, eğitim makinesi silindi", flush=True)
        elif "HATALI_BITTI" in satir:
            print("eğitim HATAYLA bitmiş — makine silinmedi; cikti/kiralik/egit-son.log", flush=True)
        else:
            print("eğitim bitmemiş görünüyor — makine açık bırakıldı (egit-izle ile izlenebilir)", flush=True)
        return


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("islem", choices=["paketle", "ac", "baslat", "durum", "indir", "kapat", "izle",
                                      "egit", "egit-durum", "egit-indir", "egit-izle",
                                      "plbert-ac", "plbert-baslat", "plbert-izle", "plbert-kurtar",
                                      "plbert-gonder", "egit-kurtar"])
    ap.add_argument("--gpu", default="", help="ac: virgülle GPU türleri (boş = GPULAR)")
    ap.add_argument("--epoch", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=300)
    ap.add_argument("--devam", default="", help="egit: sürdürülecek Models/Turkce/<.pth>")
    ap.add_argument("--veri-yok", action="store_true", help="egit: veri zaten makinede")
    ap.add_argument("--dosya", default="")
    ap.add_argument("--adet", type=int, default=1)
    ap.add_argument("--guvenli", action="store_true", help="doğrudan Secure Cloud (pahalı, daha güvenilir)")
    ap.add_argument("--sesler", default="")
    ap.add_argument("--sinir", type=int, default=0)
    ap.add_argument("--aday", type=int, default=2)
    ap.add_argument("--isci", type=int, default=3, help="makine başına yan yana uret.py sayısı")
    ap.add_argument("--pod", default=None)
    ap.add_argument("--adim", type=int, default=100_000, help="plbert-baslat: eğitim adımı")
    a = ap.parse_args()
    if a.gpu:
        GPULAR[:] = [g.strip() for g in a.gpu.split(",")]
    {"paketle": paketle,
     "ac": lambda: ac(a.adet, a.guvenli, "uretim"),
     "baslat": lambda: baslat([s for s in a.sesler.split(",") if s], a.sinir, a.aday, a.isci, a.pod),
     "durum": durum, "indir": lambda: indir(a.pod), "kapat": lambda: kapat(a.pod), "izle": izle,
     "egit": lambda: egit_baslat(a.epoch, a.batch, a.maxlen, a.devam, not a.veri_yok),
     "egit-durum": egit_durum, "egit-indir": lambda: egit_indir(a.dosya), "egit-izle": egit_izle, "egit-kurtar": egit_kurtar,
     **{f"plbert-{k}": (lambda k=k: getattr(__import__("plbert"), f"plbert_{k}")()) for k in ("ac", "izle", "kurtar")},
     "plbert-baslat": lambda: __import__("plbert").plbert_baslat(a.adim, a.batch),
     "plbert-gonder": lambda: __import__("plbert").plbert_egitime_gonder()}[a.islem]()
