# -*- coding: utf-8 -*-
"""
Kaggle işleri — öğretmen verisini Kaggle'ın ücretsiz GPU'sunda üretmek.

Kimlik: Kaggle → Your API tokens → token'ı C:\\Users\\<sen>\\.kaggle\\access_token
dosyasına kaydet (sohbete / depoya YAZMA).

    python bulut/kaggle/kaggleIs.py paketle            girdi veri setini (özel) yükler / günceller
    python bulut/kaggle/kaggleIs.py gonder 0 20        korpus[0:20] üretimini başlatır (deneme)
    python bulut/kaggle/kaggleIs.py gonder 20 1500     büyük parti (oturum ≤ ~11,5 sa)
    python bulut/kaggle/kaggleIs.py gonder 3100 4600 --cekirdek sesatolye-uretim-2 --aday 2 --duygu --merak-ek 300
                                                       duygu etiketli, 2 adaylı parti (ayrı çekirdek)
    python bulut/kaggle/kaggleIs.py durum [--cekirdek …] çalışıyor mu, bitti mi
    python bulut/kaggle/kaggleIs.py indir              çıktıyı cikti/veri/ ile birleştirir
    python bulut/kaggle/kaggleIs.py tasarim [--adaylar ayse-1,…]   kadro adaylarını üretir (tasarim.py)
    python bulut/kaggle/kaggleIs.py tasarim-indir      adayları cikti/adaylar/ altına indirir
    python bulut/kaggle/kaggleIs.py tazele [--tasarla serdar-4,…]  seçilen seslerin kimlik tazelemesi + A/B/C
    python bulut/kaggle/kaggleIs.py tazele-indir       → cikti/tazele/ ; sayfa: ogretmen/tazeleYap.py
    python bulut/kaggle/kaggleIs.py yeniden            veri/plan_yeniden planlarını okutur (veri seti + her plana çekirdek)
    python bulut/kaggle/kaggleIs.py yeniden-durum      çekirdekler çalışıyor mu
    python bulut/kaggle/kaggleIs.py yeniden-indir      → cikti/kaggle/indirilen/ (sonra bulut/kiralik/yenile.py birlestir)
    python bulut/kaggle/kaggleIs.py dene | dene-indir  öğretmen denemesi (Chatterbox / VoxCPM2 referans + tarif)
    python bulut/kaggle/kaggleIs.py iki1 | iki1-indir | iki2 | iki2-indir [--tur tum]   iki aşamalı yapay ses
                                                       (Chatterbox → VoxCPM2); tum: bütün sesler

Kaggle'a giden: derlem metni (CC0 + bizim cümleler), Damla/Kuzey referans
kaydı (tariften tasarlanmış, kimseye ait değil), ölçü kodu. Veri seti ÖZEL.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[2]
BURASI = Path(__file__).resolve().parent
IS = KOK / "cikti" / "kaggle"
# kaggle.exe değil: başlatıcı .exe kurulduğu yolu gömüyor, klasör taşınınca (07.10 sesatolye → seslen) bozuldu
KAGGLE = [str(KOK / ".venv" / "Scripts" / "python.exe"), "-c", "from kaggle.cli import main; main()"]
SESLER = {"damla": "damla-4", "kuzey": "kuzey-1"}      # ogretmen/veriUret.py ile aynı seçim
VERI_SETI = "sesatolye-girdi"
CEKIRDEK = "sesatolye-uretim"
TASARIM = "sesatolye-tasarim"
TAZELE = "sesatolye-tazele"
GOZDEN = ["serdar-1"]           # seçilmedi ama yeni adaylarla karşılaştırılacak
sys.path.insert(0, str(KOK / "ogretmen"))
import kadro  # noqa: E402


def kaggle(*a: str, yakala: bool = False) -> str:
    # UTF-8: Kaggle günlükteki Türkçe karakterleri yazarken Windows konsolunun
    # cp1254 kodlamasında "charmap" hatasıyla düşüyordu (indirme bitse bile).
    ortam = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    r = subprocess.run([*KAGGLE, *a], capture_output=yakala, text=True, encoding="utf-8", env=ortam)
    if r.returncode:
        raise SystemExit(f"kaggle {' '.join(a)} başarısız:\n{(r.stderr or '') if yakala else ''}")
    return r.stdout if yakala else ""


def kullanici() -> str:
    cikti = kaggle("config", "view", yakala=True)
    for s in cikti.splitlines():
        if s.strip().startswith("- username:"):
            return s.split(":", 1)[1].strip()
    raise SystemExit("Kaggle kullanıcı adı okunamadı: token ~/.kaggle/access_token'da mı?")


def paketle() -> None:
    d = IS / "girdi"
    if d.exists():
        shutil.rmtree(d)
    (d / "ref").mkdir(parents=True)
    shutil.copy(KOK / "veri" / "korpus.jsonl", d)
    shutil.copy(KOK / "araclar" / "olcu" / "olcu.py", d)
    shutil.copy(KOK / "ogretmen" / "duygu.py", d)
    shutil.copy(KOK / "ogretmen" / "kadro.py", d)
    for kar, aday in SESLER.items():
        shutil.copy(KOK / "cikti" / "adaylar" / aday / "ref.wav", d / "ref" / f"{kar}.wav")
    # kadronun seçilen (ve gözden geçirilen) adaylarının kimlik kayıtları: aday/<aday>.wav
    (d / "aday").mkdir()
    for aday in {*kadro.secilenler().values(), *GOZDEN}:
        kaynak = KOK / "cikti" / "adaylar" / aday / "ref.wav"
        if not kaynak.exists():                  # 2. tur adayları (serdar-4) yalnız tazele çıktısında
            kaynak = KOK / "cikti" / "tazele" / aday / "ref.wav"
        shutil.copy(kaynak, d / "aday" / f"{aday}.wav")
    kul = kullanici()
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": VERI_SETI, "id": f"{kul}/{VERI_SETI}", "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
    var = VERI_SETI in kaggle("datasets", "list", "--mine", "-s", VERI_SETI, yakala=True)
    if var:
        kaggle("datasets", "version", "-p", str(d), "-m", "guncelleme", "-r", "zip")
    else:
        kaggle("datasets", "create", "-p", str(d), "-r", "zip")     # varsayılan: özel
    print(f"veri seti hazır: {kul}/{VERI_SETI} (özel)")


def gonder(bas: int, son: int, cekirdek: str = CEKIRDEK, adim: int = 6, sure_dk: int = 690,
           aday: int = 1, duygu: bool = False, merak_ek: int = 0) -> None:
    ayar = {"bas": bas, "son": son, "adim": adim, "sesler": list(SESLER), "sure_dk": sure_dk,
            "aday": aday, "duygu": duygu, "merak_ek": merak_ek}
    it(cekirdek, "uret.py", ayar)


def it(cekirdek: str, betik: str, ayar: dict, veri_seti: str = VERI_SETI) -> None:
    """bulut/kaggle/<betik>'i başına AYAR satırı ekleyerek çekirdeğe gönderir."""
    d = IS / "cekirdek" / cekirdek
    d.mkdir(parents=True, exist_ok=True)
    kul = kullanici()
    kod = (BURASI / betik).read_text(encoding="utf-8")
    # Python değişmezi (repr): json.dumps "true" yazıyordu → Kaggle'da NameError.
    (d / betik).write_text(f"AYAR = {ayar!r}\n" + kod, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{kul}/{cekirdek}", "title": cekirdek, "code_file": betik, "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{kul}/{veri_seti}"], "competition_sources": [], "kernel_sources": []}),
        encoding="utf-8")
    kaggle("kernels", "push", "-p", str(d))
    (IS / f"gonderim-{cekirdek}.json").write_text(json.dumps(ayar), encoding="utf-8")
    print(f"gönderildi: {ayar} → kaggle.com/code/{kul}/{cekirdek}")


YENIDEN = "seslen-yeniden"            # veri seti; çekirdekler seslen-yeniden-uretim-<N> (aynı ad olamıyor: 409)
PLAN_YENIDEN = KOK / "veri" / "plan_yeniden"


def yeniden() -> None:
    """Yeniden üretim (bulut/kiralik/yenile.py plan → veri/plan_yeniden) Kaggle'da, ücretsiz: her plan<N>.jsonl ayrı
    çekirdek. refler.json: {ses: [ref.wav, ref2.wav ya da null]}; ayar.json: yeniden.py AYAR'ı (parca hariç)."""
    import time
    d = IS / "yeniden"
    if d.exists():
        shutil.rmtree(d)
    (d / "ref").mkdir(parents=True)
    planlar = sorted(PLAN_YENIDEN.glob("plan*.jsonl"))
    if not planlar:
        raise SystemExit("veri/plan_yeniden yok: önce python bulut/kiralik/yenile.py plan")
    for f in planlar:
        shutil.copy(f, d)
    for ses, (ref, ref2) in json.loads((PLAN_YENIDEN / "refler.json").read_text(encoding="utf-8")).items():
        shutil.copy(KOK / ref, d / "ref" / f"{ses}.wav")
        if ref2:
            shutil.copy(KOK / ref2, d / "ref" / f"{ses}2.wav")
    for f in (KOK / "araclar" / "olcu" / "olcu.py", KOK / "araclar" / "olcu" / "denetim.py", KOK / "ogretmen" / "duygu.py",
              KOK / "ogretmen" / "kadro.py"):
        shutil.copy(f, d)
    kul = kullanici()
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": YENIDEN, "id": f"{kul}/{YENIDEN}", "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
    if YENIDEN in kaggle("datasets", "list", "--mine", "-s", YENIDEN, yakala=True):
        kaggle("datasets", "version", "-p", str(d), "-m", "guncelleme", "-r", "zip")
    else:
        kaggle("datasets", "create", "-p", str(d), "-r", "zip")     # varsayılan: özel
    print(f"veri seti yüklendi: {kul}/{YENIDEN} (özel)", flush=True)
    for _ in range(60):                                # işlenmeden başlayan çekirdek eski girdiyi görür
        if kaggle("datasets", "status", f"{kul}/{YENIDEN}", yakala=True).strip().endswith("ready"):
            break
        time.sleep(10)
    ayar = json.loads((PLAN_YENIDEN / "ayar.json").read_text(encoding="utf-8"))
    for f in planlar:
        n = int(f.stem.removeprefix("plan"))
        it(f"{YENIDEN}-uretim-{n}", "yeniden.py", {**ayar, "parca": n}, YENIDEN)


def yeniden_durum() -> None:
    for f in sorted(PLAN_YENIDEN.glob("plan*.jsonl")):
        durum(f"{YENIDEN}-uretim-{f.stem.removeprefix('plan')}")


def cikti_indir(cekirdek: str, hedef: Path, is_: int = 16) -> int:
    """Çekirdek çıktısını hızlı indirir: kaggle CLI her dosyayı sırayla ve var olanı da yeniden isteyerek çekiyordu
    (10.10: 4.900 dosyada saatler sürdü, bilgisayar uyuyunca takıldı). Liste 500'lük sayfalarla (1000 → 400 hatası), yerelde olan
    ses dosyaları atlanır (olcum/ref/log her seferinde), eksikler paralel. İndirilen dosya sayısını döndürür."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api = KaggleApi()
    api.authenticate()
    ogeler, token, log = [], None, ""
    with api.build_kaggle_client() as k:
        while True:
            istek = ApiListKernelSessionOutputRequest()
            istek.user_name, istek.kernel_slug = kullanici(), cekirdek
            api._set_paging(istek, 500, token)
            yanit = k.kernels.kernels_api_client.list_kernel_session_output(istek)
            ogeler += [(o.file_name, o.url) for o in (yanit.files or [])]
            log = log or (yanit.log or "")
            token = yanit.next_page_token
            if not token:
                break
    if log:
        (hedef / f"{cekirdek}.log").write_text(log, encoding="utf-8")
    eksik = [(a, u) for a, u in ogeler if not a.endswith(".flac") or not (hedef / a).exists()]

    def al(o: tuple[str, str]) -> None:
        yol = hedef / o[0]
        yol.parent.mkdir(parents=True, exist_ok=True)
        for deneme in range(4):
            try:
                v = requests.get(o[1], timeout=120)
                v.raise_for_status()
                gecici = yol.with_name(yol.name + ".parca")
                gecici.write_bytes(v.content)
                gecici.replace(yol)
                return
            except Exception:
                if deneme == 3:
                    raise
    with ThreadPoolExecutor(is_) as ex:
        list(ex.map(al, eksik))
    print(f"{cekirdek}: {len(ogeler)} dosya, {len(eksik)} indirildi", flush=True)
    return len(eksik)


def yeniden_indir() -> None:
    """Çekirdek çıktıları → cikti/kaggle/indirilen/<çekirdek>/ (veri/<ses>/, ref/); birleştirme yenile.py'de."""
    for f in sorted(PLAN_YENIDEN.glob("plan*.jsonl")):
        cek = f"{YENIDEN}-uretim-{f.stem.removeprefix('plan')}"
        hedef = IS / "indirilen" / cek
        hedef.mkdir(parents=True, exist_ok=True)       # silinmez: yarıda kalırsa inenler yeniden inmez
        cikti_indir(cek, hedef)
        for o in sorted(hedef.glob("veri/*/olcum.jsonl")):
            satir = o.read_text(encoding="utf-8").splitlines()
            print(f"{cek} {o.parent.name}: {len(satir)} kayıt, kapıdan geçen {sum(json.loads(x)['gecti'] for x in satir)}")
        log = list(hedef.glob("*.log"))
        for s in (log[0].read_text(encoding="utf-8", errors="replace").splitlines() if log else []):
            if "ÖZET" in s or "GPU:" in s or "süre doldu" in s or "ref2" in s:
                print(s[-200:])


DENE = "seslen-dene"                   # 13.10 Hasan aksan denemesi: veri seti; çekirdekler seslen-dene-<motor>
YASLI = ("A very old man in his late seventies, aged voice with a slight tremble and gentle rasp, speaking slowly and "
         "calmly like a wise grandfather")
DENE_ADAYLAR = {
    "chatterbox": {"A1_chatterbox_hasan6_aksansiz": {"ref": "hasan6", "cfg": 0.0},
                   "A2_chatterbox_hasan6": {"ref": "hasan6", "cfg": 0.5},
                   "A3_chatterbox_hasan5_aksansiz": {"ref": "hasan5", "cfg": 0.0}},
    "voxcpm": {"B1_kuzey_yasli": {"ref": "kuzey", "tarif": YASLI},
               "B2_serdar_yasli": {"ref": "serdar", "tarif": YASLI, "cfg": 2.5},
               "C1_turkce_tarif": {"tarif": "Yetmiş beş yaşında, Anadolulu, bilge bir dede; kalın ve yaşlı, hafif "
                                            "hırıltılı bir sesle, yavaş ve sakin konuşuyor."},
               "C2_turkce_tarif": {"tarif": "Seksen yaşlarında yaşlı bir Türk erkeği; derin, titrek ve sıcak bir sesle "
                                            "torunlarına masal anlatıyor."},
               "D1_youtube1_yasli": {"ref": "yodas1", "tarif": YASLI},
               "D2_youtube2_yasli": {"ref": "yodas2", "tarif": YASLI},
               "D3_youtube3_yasli": {"ref": "yodas3", "tarif": YASLI}},
}


# 13.10 iki aşamalı yapay ses: 1) Chatterbox (dil tr, referans aksanı kapalı) tasarım seslerimizin rengiyle uzun
# paragraf → 2) VoxCPM2 bunu referans alıp tarifle yaşlandırır (D yöntemi, gerçek kişi olmadan)
PARAGRAF = ("Evladım, gel otur şöyle yanıma. Sana gençliğimden bir hikâye anlatayım. O zamanlar köyümüzde ne elektrik "
            "vardı ne de su; ama insanlar birbirine çok yakındı, akşamları hep birlikte oturur, sohbet ederdik.")
IKI1 = {"chatterbox": {f"iki_{r}": {"ref": r, "cfg": 0.0, "abarti": 0.5, "metin": PARAGRAF}
                       for r in ("hasan6", "hasan5", "kuzey", "serdar")}}
IKI2 = {"voxcpm": {**{f"H_{r}_yasli": {"ref": f"iki_{r}", "tarif": YASLI} for r in ("hasan6", "hasan5", "kuzey", "serdar")},
                   "H_kuzey_yasli_cfg3": {"ref": "iki_kuzey", "tarif": YASLI, "cfg": 3.0}}}


# Bütün sesler (13.10): aynı iki aşama; 2. aşamanın tarifi sesin bugünkü karakteri (kadrodaki tarif bazı seslerde
# başka sese aitti: Zeynep emre-3'ün "genç erkek" tarifiyle doğmuştu). Hasan ayrı yapıldı (IKI1/IKI2).
NOTR_PARAGRAF = ("Bugün hava çok güzeldi. Sabah erkenden kalktım, kahvaltımı yaptım ve parka yürüyüşe çıktım. Yolda "
                 "eski bir arkadaşıma rastladım; uzun uzun sohbet ettik, sonra birlikte birer çay içtik.")
TUM_TARIF = {
    "ayse": "A friendly mature woman around forty five, warm medium-low voice, motherly and reassuring",
    "serdar": "A mature man in his mid forties, deep bass-baritone voice, clear and smooth, authoritative yet friendly",
    "zeynep": "A teenage girl around seventeen, light bright voice, lively, cheerful and quick",
    "emre": "A teenage boy around eighteen, medium bright voice, energetic, casual and friendly",
    "selin": "A young woman in her twenties, soft gentle voice, warm and kind",
    "fatma": "An elderly village grandmother in her late seventies, low-pitched raspy old woman's voice, warm and loving",
    "elif": "A little girl about eight years old, high sweet child voice, curious and cheerful",
    "can": "A young boy about nine years old, bright child voice, excited and talkative",
    "ece": "A small girl, very thin high voice, playful and giggly",
    "pitircik": "A cute tiny cartoon creature, very thin squeaky high voice, excited and adorable",
    "koca": "A large clumsy cartoon giant, thick rumbling low voice, warm and silly",
    "firfir": "A cunning cartoon fox, smooth sly voice with a nasal twist, witty and flirtatious",
    "kara": "A cartoon villain wizard, low raspy sinister voice, dramatic and theatrical",
}
TUM_TARIF = {**kadro.OZEL.get("tarif", {}), **TUM_TARIF}       # ürün sesleri (Damla, Kuzey): kadro._ozel
TUM1 = {"chatterbox": {f"iki_{v}": {"ref": v, "cfg": 0.0, "abarti": 0.5, "metin": NOTR_PARAGRAF} for v in TUM_TARIF}}
TUM2 = {"voxcpm": {f"{v}_iki": {"ref": f"iki_{v}", "tarif": t} for v, t in TUM_TARIF.items()}}
IKI_TUR = {"hasan": (IKI1, IKI2, "hasan_iki"), "tum": (TUM1, TUM2, "tum_iki")}

# 15.10 Can: seçim sayfasında bütün Can'lar "genç kız gibi". Erkek tınısı Emre'nin 1. yol paragrafından (iki_emre:
# Chatterbox Türkçesi) gelir, tarif yaşı küçültür; ayrıca yalnız tariften iki tasarım.
OGLAN = ("A nine-year-old boy, clearly a boy's voice and not a girl's, a little husky and lower than a girl, "
         "energetic, cheeky and talkative")
OGLAN2 = ("A ten-year-old schoolboy whose voice has not broken yet but is unmistakably boyish, slightly rough and loud, "
          "playful and excited")
# 20.10 Can: A sesi (255 Hz) 9 yaşına kalın bulundu; A'nın 3 yarım ton inceltilmiş kaydı (dene/ref/can_a_p3.wav) + bu tarif
# (aday G, 316 Hz; bulut/modal/ogretmenModal.py ile üretildi, cikti/teshis/can_dene2)
OGLAN8 = "An eight-year-old boy, small child's voice, high but boyish, playful and energetic"
CAN = {"voxcpm": {"can_emre_oglan": {"ref": "iki_emre", "tarif": OGLAN},
                  "can_emre_oglan_cfg3": {"ref": "iki_emre", "tarif": OGLAN, "cfg": 3.0},
                  "can_emre_oglan2": {"ref": "iki_emre", "tarif": OGLAN2},
                  "can_tasarim_oglan": {"tarif": OGLAN},
                  "can_tasarim_oglan2": {"tarif": OGLAN2},
                  "can_turkce_tarif": {"tarif": "Dokuz yaşında, yaramaz ve neşeli bir erkek çocuk; sesi kız sesinden "
                                                "kalınca, hafif boğuk, hızlı ve heyecanlı konuşuyor"}}}


def tum_hazirla() -> None:
    """Her sesin kimlik kaydı (cikti/tazele/<aday>/ref2.wav) → dene/ref/<ses>.wav; karşılaştırma için şimdiki kayıt."""
    import soundfile as sf
    yer = KOK / "cikti" / "teshis" / "tum_iki"
    yer.mkdir(parents=True, exist_ok=True)
    for ses in TUM_TARIF:
        r2 = KOK / "cikti" / "tazele" / kadro.secilenler()[ses] / "ref2.wav"
        shutil.copy(r2, IS / "dene" / "ref" / f"{ses}.wav")
        x, sr = sf.read(r2)
        sf.write(yer / f"{ses}_0_simdiki.wav", x, sr)
    print(f"{len(TUM_TARIF)} ses hazır", flush=True)


def dene(adaylar: dict = DENE_ADAYLAR) -> None:
    """Öğretmen denemesi (ogretmenDene.py): cikti/kaggle/dene/ref/*.wav + kadro + olcu → veri seti, her motora çekirdek."""
    import time
    d = IS / "dene"
    for f in (KOK / "ogretmen" / "kadro.py", KOK / "araclar" / "olcu" / "olcu.py"):
        shutil.copy(f, d)
    kul = kullanici()
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": DENE, "id": f"{kul}/{DENE}", "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
    if DENE in kaggle("datasets", "list", "--mine", "-s", DENE, yakala=True):
        kaggle("datasets", "version", "-p", str(d), "-m", "guncelleme", "-r", "zip")
    else:
        kaggle("datasets", "create", "-p", str(d), "-r", "zip")
    for _ in range(60):
        if kaggle("datasets", "status", f"{kul}/{DENE}", yakala=True).strip().endswith("ready"):
            break
        time.sleep(10)
    for motor, a in adaylar.items():
        it(f"{DENE}-{motor}", "ogretmenDene.py", {"motor": motor, "adaylar": a}, DENE)


def dene_indir(adaylar: dict = DENE_ADAYLAR, klasor: str = "hasan_dene") -> None:
    """Çıktılar → cikti/teshis/<klasor>/<aday>_<n>.wav (dinleme) + olcum özetleri. İki aşamanın 1.'sinde (IKI1)
    paragraf kaydı ayrıca 2. aşamanın referansı olur: cikti/kaggle/dene/ref/<aday>.wav."""
    import soundfile as sf
    yer = KOK / "cikti" / "teshis" / klasor
    yer.mkdir(parents=True, exist_ok=True)
    for motor in adaylar:
        hedef = IS / "indirilen" / f"{DENE}-{motor}"
        hedef.mkdir(parents=True, exist_ok=True)
        cikti_indir(f"{DENE}-{motor}", hedef)
        for o in sorted((hedef / "dene").glob("*/olcum.json")):
            if o.parent.name not in adaylar[motor]:      # klasörde önceki turların çıktıları da duruyor
                continue
            k = json.loads(o.read_text(encoding="utf-8"))
            for f in sorted(o.parent.glob("*.flac")):
                x, sr = sf.read(f)
                sf.write(yer / f"{o.parent.name}_{int(f.stem) + 1}.wav", x, sr)
                if adaylar in (IKI1, TUM1):
                    sf.write(IS / "dene" / "ref" / f"{o.parent.name}.wav", x, sr)
            print(f"{o.parent.name}: doğruluk {[m['dogruluk'] for m in k['olcum']]}, "
                  f"netlik ort {sum(m['netlik'] for m in k['olcum']) / len(k['olcum']):.1f}")


def tasarim_indir(cekirdek: str = TASARIM, alt: str = "tasarim", yer_kok: str = "adaylar") -> None:
    """Çekirdek çıktısındaki <alt>/<aday>/ klasörlerini cikti/<yer_kok>/<aday>/ altına koyar
    (varsa üzerine yazar). tasarim → cikti/adaylar, tazele → cikti/tazele."""
    hedef = IS / "indirilen" / cekirdek
    if hedef.exists():
        shutil.rmtree(hedef)
    hedef.mkdir(parents=True)
    kaggle("kernels", "output", f"{kullanici()}/{cekirdek}", "-p", str(hedef))
    n = 0
    for d in sorted((hedef / alt).glob("*/olcum.json")):
        yer = KOK / "cikti" / yer_kok / d.parent.name
        if yer.exists():
            shutil.rmtree(yer)
        shutil.copytree(d.parent, yer)
        n += 1
    print(f"{n} aday indirildi → cikti/{yer_kok}/")


def durum(cekirdek: str = CEKIRDEK) -> None:
    print(kaggle("kernels", "status", f"{kullanici()}/{cekirdek}", yakala=True).strip())


def indir(cekirdek: str = CEKIRDEK) -> None:
    hedef = IS / "indirilen" / cekirdek
    if hedef.exists():
        shutil.rmtree(hedef)
    hedef.mkdir(parents=True)
    kaggle("kernels", "output", f"{kullanici()}/{cekirdek}", "-p", str(hedef))
    import soundfile as sf
    for kar in SESLER:
        kaynak = hedef / "veri" / kar
        if not (kaynak / "olcum.jsonl").exists():
            continue
        yer = KOK / "cikti" / "veri" / kar
        (yer / "wavs").mkdir(parents=True, exist_ok=True)
        (yer / "adaylar").mkdir(parents=True, exist_ok=True)
        eklenen = 0
        with open(yer / "olcum.jsonl", "a", encoding="utf-8") as f:
            for s in (kaynak / "olcum.jsonl").read_text(encoding="utf-8").splitlines():
                o = json.loads(s)
                wav = yer / "wavs" / f"{o['kimlik']}.wav"
                if wav.exists():
                    continue
                x, sr = sf.read(kaynak / "wavs" / f"{o['kimlik']}.flac", dtype="int16")
                sf.write(wav, x, sr, subtype="PCM_16")
                # seçilmeyen adaylar (zevk modeli / oylama için) flac olarak kalır
                for a in kaynak.glob(f"wavs/{o['kimlik']}.a*.flac"):
                    shutil.copy(a, yer / "adaylar" / a.name)
                o["yer"] = f"kaggle:{cekirdek}"
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
                eklenen += 1
        print(f"{kar}: {eklenen} kayıt eklendi")
    log = list(hedef.glob("*.log"))
    for s in (log[0].read_text(encoding="utf-8", errors="replace").splitlines() if log else []):
        if "ÖZET" in s or "GPU:" in s or "dtype" in s or "süre doldu" in s:
            print(s[-200:])
    sys.path.insert(0, str(KOK / "ogretmen"))
    from veriUret import metadata_yaz  # noqa: E402
    for kar in SESLER:
        if (KOK / "cikti" / "veri" / kar / "olcum.jsonl").exists():
            print(f"{kar}: metadata {metadata_yaz(kar)} kayıt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("islem", choices=["paketle", "gonder", "durum", "indir", "tasarim", "tasarim-indir",
                                      "tazele", "tazele-indir", "yeniden", "yeniden-durum", "yeniden-indir", "dene", "dene-indir", "iki1", "iki1-indir", "iki2", "iki2-indir", "can", "can-indir"], nargs="?", default="durum")
    ap.add_argument("--tasarla", default="", help="tazele: önce tariften tasarlanacak adaylar (virgülle)")
    ap.add_argument("--adaylar", default="", help="tasarim: virgülle aday adları (boş = seçilmemiş hepsi)")
    ap.add_argument("bas", type=int, nargs="?")
    ap.add_argument("son", type=int, nargs="?")
    ap.add_argument("--cekirdek", default=CEKIRDEK, help="ayrı çekirdek adı: partiler birbirini kesmesin")
    ap.add_argument("--aday", type=int, default=1, help="cümle başına okuma; en iyisi seçilir")
    ap.add_argument("--duygu", action="store_true", help="ogretmen/duygu.py ile duygu etiketli oku")
    ap.add_argument("--merak-ek", type=int, default=0, help="önceki partiden N soruyu merak ile yeniden oku")
    ap.add_argument("--sure-dk", type=int, default=690)
    ap.add_argument("--tur", default="hasan", help="iki1/iki2: hasan | tum (bütün sesler)")
    a = ap.parse_args()
    if a.islem == "paketle":
        paketle()
    elif a.islem == "gonder":
        gonder(a.bas, a.son, a.cekirdek, sure_dk=a.sure_dk, aday=a.aday, duygu=a.duygu, merak_ek=a.merak_ek)
    elif a.islem == "indir":
        indir(a.cekirdek)
    elif a.islem == "tasarim":
        it(TASARIM, "tasarim.py", {"adaylar": [x for x in a.adaylar.split(",") if x]})
    elif a.islem == "tasarim-indir":
        tasarim_indir()
    elif a.islem == "tazele":
        sesler = {**kadro.secilenler(), **{g: g for g in GOZDEN}}
        it(TAZELE, "tazele.py", {"sesler": sesler, "tasarla": [x for x in a.tasarla.split(",") if x]})
    elif a.islem == "tazele-indir":
        tasarim_indir(TAZELE, "tazele", "tazele")
    elif a.islem == "yeniden":
        yeniden()
    elif a.islem == "yeniden-durum":
        yeniden_durum()
    elif a.islem == "yeniden-indir":
        yeniden_indir()
    elif a.islem == "dene":
        dene()
    elif a.islem == "dene-indir":
        dene_indir()
    elif a.islem == "can":
        dene(CAN)
    elif a.islem == "can-indir":
        dene_indir(CAN, "can_dene")
    elif a.islem in ("iki1", "iki2"):
        if a.tur == "tum" and a.islem == "iki1":
            tum_hazirla()
        dene(IKI_TUR[a.tur][0 if a.islem == "iki1" else 1])
    elif a.islem in ("iki1-indir", "iki2-indir"):
        bir, iki, klasor = IKI_TUR[a.tur]
        dene_indir(bir if a.islem == "iki1-indir" else iki, klasor)
    else:
        durum(a.cekirdek)
