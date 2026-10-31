# -*- coding: utf-8 -*-
"""
TÜRKÇE PL-BERT EĞİTİMİ (RunPod) — kod, veri ve model ayrı projede (Masaüstü/ses_projeleri/turkce-plbert), eğitim
ayrı makinede. 18.10'da runpod.py 700 satıra dayanınca oradan ayrıldı; komutlar yine runpod.py'den:
    python bulut/kiralik/runpod.py plbert-ac | plbert-baslat | plbert-izle | plbert-kurtar | plbert-gonder
"""
from runpod import *  # noqa: F401,F403  (KOK, ssh, podlar, ac … ve runpod'un içe aktardığı modüller)
from runpod import _egitim_podu, _graphql  # noqa: F401

PLBERT = Path(os.environ.get("TURKCE_PLBERT", Path.home() / "Desktop" / "ses_projeleri" / "turkce-plbert"))
def _plbert_podu() -> dict:
    return next(x for x in podlar() if x.get("rol") == "plbert")


def plbert_ac() -> None:
    ac(1, guvenli=True)
    p = podlar()
    p[-1]["rol"] = "plbert"
    podlar_yaz(p)


def plbert_baslat(adim: int, batch: int) -> None:
    """Kod + veri (turkce-plbert/veri: tr/, token_maps.pkl, ml/) → /workspace/plbert, plbert.sh başlar."""
    pod = _plbert_podu()
    kok = PLBERT / "veri"
    dosyalar = [(p, f"plbert/{p.relative_to(kok).as_posix()}") for p in sorted((kok / "tr").rglob("*")) if p.is_file()]
    dosyalar += [(kok / "token_maps.pkl", "plbert/token_maps.pkl"), (kok / "ml" / "step_1100000.t7", "plbert/ml/step_1100000.t7")]
    dosyalar += [(PLBERT / f, f"plbert/{f}") for f in ("egit.py", "config_tr.yml")]
    dosyalar += [(PLBERT / "PL-BERT" / f, f"plbert/PL-BERT/{f}")
                 for f in ("dataloader.py", "model.py", "utils.py", "text_utils.py")]
    dosyalar += [(PLBERT / "plbert.sh", "plbert.sh")]
    print(f"gönderiliyor: {sum(p.stat().st_size for p, _ in dosyalar) / 1e6:.0f} MB", flush=True)
    t0 = time.time()
    s = subprocess.Popen(["ssh", *ssh_arg(pod), f"root@{pod['ip']}", "rm -rf /workspace/plbert/tr && tar xf - -C /workspace"],
                         stdin=subprocess.PIPE)
    with tarfile.open(fileobj=s.stdin, mode="w|") as t:
        for p, ad in dosyalar:
            t.add(p, arcname=ad)
    s.stdin.close()
    if s.wait():
        raise SystemExit("gönderilemedi")
    print(f"gönderildi ({(time.time() - t0) / 60:.1f} dk)", flush=True)
    # PL-BERT bitince 120 dk bekler: durmuş makine yeniden açılamayabiliyor (02.10, host'ta boş GPU yok),
    # yerel izleyici (plbert-izle) sonucu bu sürede indirip makineyi siler.
    # Windows'ta düzenlenen betik CRLF olabiliyor; bash satır sonundaki CR yüzünden hiç başlamıyordu (02.10).
    r = ssh(pod, f"sed -i 's/\\r$//' /workspace/plbert.sh && cd /workspace && (ADIM={adim} BATCH={batch} GRACE=120 "
                 f"setsid nohup bash plbert.sh < /dev/null > /dev/null 2>&1 &)", timeout=120)
    print(f"{pod['ad']}: PL-BERT başladı ({adim} adım, batch {batch})" + ("" if r.returncode == 0 else f" HATA {r.stderr[-300:]}"))


def plbert_izle(aralik_dk: int = 10) -> None:
    """Kaydı cikti/kiralik/plbert-son.log'a kopyalar; başarıyla bitince son ağırlığı indirip makineyi siler."""
    hedef = PLBERT / "model"
    hedef.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            pod = _plbert_podu()
            durum = api(f"/pods/{pod['id']}").get("desiredStatus")
            if durum != "RUNNING":
                print(f"{time.strftime('%H:%M')} PL-BERT makinesi {durum}: elle bakılmalı", flush=True)
                return
            o = ssh(pod, "tail -n 200 /workspace/plbert.log; test -f /workspace/PLBERT_BITTI && ! pgrep -f '[e]git.py' "
                         "> /dev/null && { grep -q 'çıkış 0)' /workspace/PLBERT_BITTI && echo BITTI || echo HATALI_BITTI; }",
                    timeout=60).stdout
        except (subprocess.TimeoutExpired, OSError, StopIteration, KeyError) as h:
            print(f"{time.strftime('%H:%M')} geçici hata: {type(h).__name__}", flush=True)
            time.sleep(aralik_dk * 60)
            continue
        (IS / "plbert-son.log").write_text(o, encoding="utf-8")
        son = [x for x in o.splitlines() if x.startswith(("Step", "DOĞRULAMA"))][-1:]
        print(f"{time.strftime('%H:%M')} {son[0][:120] if son else ''}", flush=True)
        if o.rstrip().endswith("HATALI_BITTI"):
            print("PL-BERT HATAYLA bitti — makine silinmedi; plbert-son.log", flush=True)
            return
        if o.rstrip().endswith("BITTI"):
            ad = ssh(pod, "ls -t /workspace/plbert/Checkpoint_tr/step_*.t7 | head -1", timeout=60).stdout.strip()
            for uzak, yerel in ((ad, ad.rsplit("/", 1)[-1]), ("/workspace/plbert/Checkpoint_tr/config.yml", "config.yml"),
                                ("/workspace/plbert/token_maps.pkl", "token_maps.pkl")):
                with open(hedef / yerel, "wb") as f:
                    subprocess.run(["ssh", *ssh_arg(pod), f"root@{pod['ip']}", f"cat {uzak}"], stdout=f, check=True)
            kapat(pod["id"])
            print(f"PL-BERT bitti: {hedef}, makine silindi", flush=True)
            if api(f"/pods/{_egitim_podu()['id']}").get("desiredStatus") == "RUNNING":
                plbert_egitime_gonder()
            else:           # 03.10: eğitim makinesi durdurulmuştu; açılınca elle: runpod.py plbert-gonder
                print("eğitim makinesi kapalı: PL-BERT elle gönderilecek (plbert-gonder)", flush=True)
            return
        time.sleep(aralik_dk * 60)


def plbert_egitime_gonder() -> None:
    """İndirilen Türkçe PL-BERT'i öğrenci makinesine (Utils/PLBERT_tr) gönderir ve UC_HAZIR koyar:
    asama23.sh 2. aşamadan sonra bunu bekleyip 3. aşamayı (vurgu + Türkçe PL-BERT) başlatır."""
    kaynak = PLBERT / "model"
    adim = sorted(kaynak.glob("step_*.t7"), key=lambda p: int(p.stem.split("_")[1]))
    if not adim:
        raise SystemExit(f"{kaynak} içinde step_*.t7 yok")
    pod = _egitim_podu()
    s = subprocess.Popen(["ssh", *ssh_arg(pod), f"root@{pod['ip']}",
                          "rm -rf /workspace/StyleTTS2/Utils/PLBERT_tr && mkdir -p /workspace/StyleTTS2/Utils/PLBERT_tr && "
                          "tar xf - --no-same-owner -C /workspace/StyleTTS2/Utils/PLBERT_tr && touch /workspace/UC_HAZIR && "
                          "ls -la /workspace/StyleTTS2/Utils/PLBERT_tr"], stdin=subprocess.PIPE)
    with tarfile.open(fileobj=s.stdin, mode="w|") as t:
        for f in (adim[-1], kaynak / "config.yml"):
            t.add(f, arcname=f.name)
    s.stdin.close()
    if s.wait():
        raise SystemExit("PL-BERT öğrenci makinesine gönderilemedi")
    print(f"{time.strftime('%H:%M')} PL-BERT ({adim[-1].name}) öğrenci makinesinde, UC_HAZIR konuldu", flush=True)

def plbert_kurtar(aralik_dk: int = 10) -> None:
    """Kendini durdurmuş PL-BERT makinesini (sonuç indirilmeden) GPU boşalınca açar, sonucu indirir, siler.
    02.10: bilgisayar uyurken eğitim bitti, makine durdu; yeniden açılışta "host'ta boş GPU yok"."""
    hedef = PLBERT / "model"
    hedef.mkdir(parents=True, exist_ok=True)
    pod = _plbert_podu()
    while True:
        try:
            d = api(f"/pods/{pod['id']}")
            if d.get("desiredStatus") != "RUNNING":
                api(f"/pods/{pod['id']}/start", "POST")
        except SystemExit as e:
            print(f"{time.strftime('%H:%M')} açılamadı: {str(e)[-90:]}", flush=True)
            time.sleep(aralik_dk * 60)
            continue
        for _ in range(30):
            d = api(f"/pods/{pod['id']}")
            if d.get("publicIp") and (d.get("portMappings") or {}).get("22"):
                break
            time.sleep(10)
        pod.update(ip=d["publicIp"], port=d["portMappings"]["22"])
        pl = podlar()
        for x in pl:
            if x["id"] == pod["id"]:
                x.update(ip=pod["ip"], port=pod["port"])
        podlar_yaz(pl)
        time.sleep(20)
        ad = ssh(pod, "ls -t /workspace/plbert/Checkpoint_tr/step_*.t7 | head -1", timeout=60).stdout.strip()
        if not ad:
            print("kayıt bulunamadı — makine silinmedi", flush=True)
            return
        for uzak, yerel in ((ad, ad.rsplit("/", 1)[-1]), ("/workspace/plbert/Checkpoint_tr/config.yml", "config.yml"),
                            ("/workspace/plbert/token_maps.pkl", "token_maps.pkl"), ("/workspace/plbert.log", "plbert.log")):
            with open(hedef / yerel, "wb") as f:
                subprocess.run(["ssh", *ssh_arg(pod), f"root@{pod['ip']}", f"cat {uzak}"], stdout=f, check=True)
        print(f"{time.strftime('%H:%M')} indirildi: {hedef} ({ad.rsplit('/', 1)[-1]})", flush=True)
        kapat(pod["id"])
        print("PL-BERT makinesi silindi", flush=True)
        return


