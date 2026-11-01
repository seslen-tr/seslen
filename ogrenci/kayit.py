# -*- coding: utf-8 -*-
"""
ÖĞRETMEN KAYDI (29.10) — kendiliğinden dolan, öğretmen kalitesinde ses kaydı. Eğitim yok.

Öğretmen (VoxCPM2, "1. yol") işlemcide gerçek zamanın ~5,5 katı yavaş: canlıda kullanılamaz, ama arka planda olur.
  1. Stüdyo bir bölüm (paragraf parçası) söyleyecekken önce bütününü, sonra cümlelerini kayıtta arar (ses ×
     cümle; ton elle seçildiyse o ton). Okuma ayarları (hız, perde, şiddet, üslup, tonlama, virgül) değişmediyse
     öğretmenin sesi anında çalar.
  2. Yoksa öğrenci okur ve cümle sıraya yazılır (iki kez istenen öne geçer).
  3. İşçi (python ogrenci/kayit.py isci) sıradaki cümleyi öğretmenle üretir: metin okunur hâle getirilir (rakam
     verilince öğretmen İngilizceye kayıyor: "enkaradaki"), her aday CTC özdenetiminden geçer (ozdenetim.dinle, en çok
     4 aday, kısa cümlede 8). Kusursuz aday ya da aynı cümlenin öğrenci okuyuşundan kötü olmayan aday kayda girer;
     geçmeyen "gecmedi" diye işaretlenir (o cümleyi öğrenci okumaya devam eder).
Asistanın cümleleri çok tekrarlandığı için konuşmanın çoğu zamanla öğretmen sesi olur. 29.10 dinleme: aynı
konuşmada öğretmen ↔ öğrenci geçişi kulakla fark edilmiyor (cikti/teshis/karma/dinle.html).

Kayıt: kullanici/ogretmen_kaydi/ (git dışında): dizin.jsonl, sira.jsonl, <ses>/<anahtar>.flac (24 kHz).
Elif kayda girmez: öğrencisi eski sesinden eğitildi, öğretmenin 1. yol Elif'i başka ses.

    python ogrenci/kayit.py isci [--bir]       sırayı işler, sıra boşken bağlanan projelerin cümlelerini
                                               (kullanici/on_kayit/*.tsv; SESLEN_KAYIT_GB, varsayılan 5) üretir.
                                               --bir: iş kalmayınca durur; yoksa bekleyip sürer
    python ogrenci/kayit.py sabit              hazır öğretmen kayıtlarını (Kaggle seslen-yeniden-uretim-8, sb###) alır
    python ogrenci/kayit.py durum
    python ogrenci/kayit.py dinle              kullanici/ogretmen_kaydi/dinle.html (kayıttaki sesleri dinleme sayfası)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import zlib
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
KAYIT = KOK / "kullanici" / "ogretmen_kaydi"
ORAN = 24000
ADAY = 4
_REF = KOK / "cikti" / "kaggle" / "dene" / "ref"
# ses → (kimlik kaydı, karakter tarifi); bulut/kiralik/yenile.py TURLAR["t9"/"t10"]["yol1"] ile aynı
OGRETMEN_SES = {s: (_REF / f"iki_{s}.wav", s) for s in
                ("ayse", "serdar", "zeynep", "emre", "selin", "fatma", "ece", "pitircik", "koca", "firfir",
                 "damla", "kuzey", "kara")}
OGRETMEN_SES |= {"hasan": (_REF / "iki_serdar.wav", "YASLI"), "can": (_REF / "can_a_p3.wav", "OGLAN8")}
# Boş vakit: sıra boşken bağlanan projelerin bildirdiği cümleler önceden üretilir: kullanici/on_kayit/*.tsv (git
# dışında). Satır: duygu<TAB>cümle; {ad} sesin adıyla dolar. "# sesler: damla, kuzey" satırı listeyi o seslerle
# sınırlar (yoksa bütün sesler). "Belki lazım olur" diye genel liste üretilmez (01.11: işlemci boşa gidiyordu).
# Kayıt SINIR_GB'a varınca boş vakit üretimi durur (sıradaki gerçek cümleler sürer). Cümle başına ~65 KB (24 kHz FLAC).
SINIR_GB = float(os.environ.get("SESLEN_KAYIT_GB", "5"))
_AD = {"ayse": "Ayşe", "pitircik": "Pıtırcık", "firfir": "Fırfır"}
ON_KAYIT = KOK / "kullanici" / "on_kayit"
# Denetim: öğretmenin adayı kusursuzsa (ceza 0) ya da aynı cümlenin öğrenci okuyuşundan kötü değilse kayda girer
# (geçmeyen cümleyi zaten öğrenci okur; 01.11: geçmeyenlerin çoğunda öğretmen doğruydu, tanıyıcı "burdayım",
# "rika" duyuyordu). Öğrenci okuyuşu çalışan stüdyodan istenir (işçiye ikinci model sığmıyor); stüdyo yoksa yalnız
# ceza 0 geçer. Kısa cümlede (≤2 sözcük) öğretmen daha çok şaşırıyor: daha çok aday.
STUDYO = os.environ.get("SESLEN_STUDYO_URL", "http://127.0.0.1:8780")
KISA_ADAY = 8
OLCUT = 2           # dizindeki "olcut" bundan küçük "gecmedi" kayıtlar yeni ölçütle bir kez daha denenir
TAVAN = 5.0         # öğrenciden kötü değilse bile bu cezanın üstü (eksik sözcük vb.) geçmez


def anahtar(ses: str, cumle: str, ton: str | None = None) -> str:
    """Ses × cümle; ton yalnız kullanıcı tonu kendisi seçtiyse (ayar ya da [işaret]) anahtara girer. Otomatik tonda
    kayıt hangi tonla okunduysa o çalar (bağlanan projenin listesindeki ton bilerek seçilmiş)."""
    return hashlib.sha1(f"{ses}|{' '.join(cumle.split())}{'|' + ton if ton else ''}".encode()).hexdigest()[:16]


class Kayit:
    """Stüdyonun tarafı: bak, yoksa sıraya yaz. dizin.jsonl değişince kendiliğinden yeniden okunur."""

    def __init__(self, kok: Path = KAYIT):
        self.kok = kok
        self._dizin: dict[str, dict] = {}
        self._zaman = 0.0
        self._sirada: dict[str, int] = {}
        yol = kok / "sira.jsonl"
        for x in (yol.read_text(encoding="utf-8").splitlines() if yol.exists() else []):
            if x:
                a = json.loads(x)["anahtar"]
                self._sirada[a] = self._sirada.get(a, 0) + 1
        self._kilit = threading.Lock()

    def _tazele(self):
        yol = self.kok / "dizin.jsonl"
        z = yol.stat().st_mtime if yol.exists() else 0.0
        if z != self._zaman:
            self._dizin = {r["anahtar"]: r for r in map(json.loads, yol.read_text(encoding="utf-8").splitlines()) if r}
            self._zaman = z

    def bul(self, ses: str, cumle: str, ton: str | None = None) -> np.ndarray | None:
        """ton None: otomatik ton, cümlenin kaydı hangi tonda olursa. ton verildiyse o tonun kaydı ya da o tonla
        okunmuş genel kayıt."""
        import soundfile as sf
        with self._kilit:
            self._tazele()
            r = self._dizin.get(anahtar(ses, cumle, ton)) if ton else None
            if not r or r.get("durum") != "tamam":
                r = self._dizin.get(anahtar(ses, cumle))
                if r and ton and r.get("ton") != ton:
                    r = None
        if not r or r.get("durum") != "tamam":
            return None
        x, _ = sf.read(self.kok / r["dosya"], dtype="float32")
        return x

    def sira_ekle(self, ses: str, ton: str, cumle: str, sabit: bool = False) -> None:
        """Öğrencinin okuduğu cümle sıraya. Aynı cümle ikinci kez istenince bir satır daha yazılır: işçi çok
        istenenleri önce üretir (dosya cümle başına en çok iki satır büyür). sabit: ton kullanıcının seçimi."""
        if ses not in OGRETMEN_SES:
            return
        a = anahtar(ses, cumle, ton if sabit else None)
        with self._kilit:
            self._tazele()
            n = self._sirada.get(a, 0)
            if a in self._dizin or n >= 2:
                return
            self._sirada[a] = n + 1
            self.kok.mkdir(parents=True, exist_ok=True)
            with open(self.kok / "sira.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"anahtar": a, "ses": ses, "ton": ton, "cumle": cumle, "sabit": sabit},
                                   ensure_ascii=False) + "\n")


def adaylar() -> list[dict]:
    """Boş vakit cümleleri, stüdyonun böldüğü gibi (akici + turkce-duygu, telaffuz sözlüğü): stüdyo birebir bulur."""
    from turkce_duygu import cumlelere_bol
    from studyo import Telaffuz, akici
    tel, liste = Telaffuz(), []
    for dosya in sorted(ON_KAYIT.glob("*.tsv")) if ON_KAYIT.exists() else []:
        satirlar = dosya.read_text(encoding="utf-8").splitlines()
        sesler = list(OGRETMEN_SES)
        for s in satirlar:
            if s.replace(" ", "").lower().startswith("#sesler:"):
                sesler = [x.strip() for x in s.split(":", 1)[1].split(",") if x.strip() in OGRETMEN_SES]
        for ses in sesler:
            for satir in satirlar:
                if not satir.strip() or satir.startswith("#") or "\t" not in satir:
                    continue
                ton, metin = satir.split("\t", 1)
                metin = metin.replace("{ad}", _AD.get(ses, ses.capitalize()))
                for c in cumlelere_bol(akici(metin)):
                    c = tel.uygula(c)
                    liste.append({"anahtar": anahtar(ses, c), "ses": ses, "ton": ton, "cumle": c,
                                  "kaynak": f"on_kayit/{dosya.stem}"})
    return liste


def boyut_gb(kok: Path = KAYIT) -> float:
    return sum(f.stat().st_size for f in kok.glob("*/*.flac")) / 1e9


def _dizine_yaz(kok: Path, r: dict) -> None:
    with open(kok / "dizin.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _oncelik_dusur() -> None:
    """İşçi canlı konuşmayı yavaşlatmasın: Windows'ta düşük öncelik, başkasında nice."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
    else:
        os.nice(10)


def isci(bir: bool = False, kok: Path = KAYIT) -> None:
    sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "ogretmen"), str(KOK / "bulut" / "kaggle")]
    from yukle import ogretmen, sureci_hazirla
    sureci_hazirla()
    _oncelik_dusur()
    import librosa
    import soundfile as sf
    import torch
    import duygu
    import kaggleIs
    from ozdenetim import Ozdenetim, kaynak
    from turkce_vurgu import normallestir
    og = ogretmen("cpu")
    sr = og.tts_model.sample_rate
    oz = Ozdenetim()
    kok.mkdir(parents=True, exist_ok=True)
    bos_vakit = adaylar()
    print(f"işçi hazır; boş vakit listesi {len(bos_vakit)} cümle", flush=True)
    while True:
        r = siradaki(kok, bos_vakit)
        if r is None:
            if bir:
                return
            time.sleep(20)
            continue
        ses, ton, cumle = r["ses"], r["ton"], r["cumle"]
        ref_wav, tarif_ad = OGRETMEN_SES[ses]
        tarif = kaggleIs.TUM_TARIF[tarif_ad] if tarif_ad in kaggleIs.TUM_TARIF else getattr(kaggleIs, tarif_ad)
        tarif = ", ".join(x for x in (tarif, duygu.TARIF.get(ton, "")) if x)
        hedef = kaynak(cumle)
        t0 = time.perf_counter()
        en_iyi = None
        for i in range(KISA_ADAY if len(hedef.split()) <= 2 else ADAY):
            torch.manual_seed(zlib.crc32(f"{r['anahtar']}|{i}".encode()))
            x = np.asarray(og.generate(text=f"({tarif}){normallestir(cumle)}", reference_wav_path=str(ref_wav),
                                       cfg_value=2.0, inference_timesteps=10), np.float32)
            x = librosa.resample(x, orig_sr=sr, target_sr=ORAN)
            ceza, _, d = oz.dinle(x, ORAN, hedef)
            if en_iyi is None or ceza < en_iyi[0]:
                en_iyi = (ceza, x, d["duyulan"], i + 1)
            if ceza == 0:
                break
        ceza, x, duyulan, deneme = en_iyi
        ogrenci = ogrenci_cezasi(oz, ses, ton, cumle, hedef) if ceza > 0 else None
        gecti = ceza == 0 or (ogrenci is not None and ceza <= ogrenci and ceza < TAVAN)
        kayit = {**r, "ceza": ceza, "ogrenci_ceza": ogrenci, "duyulan": duyulan, "deneme": deneme, "olcut": OLCUT,
                 "sn": round(time.perf_counter() - t0, 1), "zaman": time.strftime("%Y-%m-%d %H:%M")}
        if gecti:
            (kok / ses).mkdir(exist_ok=True)
            sf.write(kok / ses / f"{r['anahtar']}.flac", x, ORAN, subtype="PCM_16")
            kayit |= {"durum": "tamam", "dosya": f"{ses}/{r['anahtar']}.flac"}
        else:
            kayit["durum"] = "gecmedi"
        _dizine_yaz(kok, kayit)
        print(f"{kayit['durum']}: {ses}/{ton} {cumle[:60]} ({kayit['sn']} sn, {deneme} aday, ceza {ceza}"
              f"{'' if ogrenci is None else f', öğrenci {ogrenci}'})", flush=True)


def siradaki(kok: Path, bos_vakit: list[dict]) -> dict | None:
    """Önce gerçek sıra (çok istenen önce, sonra geliş sırası), sıra boşsa bağlanan projelerin listesi. Eski ölçütle
    geçmeyenler bir kez daha denenir."""
    dizin = {r["anahtar"]: r for r in map(json.loads, (kok / "dizin.jsonl").read_text(encoding="utf-8").splitlines())
             if r} if (kok / "dizin.jsonl").exists() else {}
    bitti = {a for a, r in dizin.items() if r.get("durum") == "tamam" or r.get("olcut", 1) >= OLCUT}
    sira = [json.loads(x) for x in (kok / "sira.jsonl").read_text(encoding="utf-8").splitlines() if x] \
        if (kok / "sira.jsonl").exists() else []
    sayi: dict[str, int] = {}
    for r in sira:
        sayi[r["anahtar"]] = sayi.get(r["anahtar"], 0) + 1
    bekleyen = list({r["anahtar"]: r for r in sira if r["anahtar"] not in bitti}.values())
    if bekleyen:
        return min(enumerate(bekleyen), key=lambda x: (-sayi[x[1]["anahtar"]], x[0]))[1]
    if boyut_gb(kok) < SINIR_GB:
        return next((r for r in bos_vakit if r["anahtar"] not in bitti), None)
    return None


def ogrenci_cezasi(oz, ses: str, ton: str, cumle: str, hedef: str) -> float | None:
    """Aynı cümleyi çalışan stüdyoya öğrenciyle (kayıtsız, özdenetimli) okutup aynı denetimden geçirir."""
    import base64
    import io
    import urllib.request
    import soundfile as sf
    govde = json.dumps({"metin": cumle, "ayar": {"ses": ses, "ton": ton, "kayit": False}}).encode()
    try:
        istek = urllib.request.Request(f"{STUDYO}/api/seslendir", govde, {"Content-Type": "application/json"})
        parca = []
        with urllib.request.urlopen(istek, timeout=300) as y:
            for satir in y:
                p = json.loads(satir)
                if p.get("tur") == "hata":
                    return None
                if p.get("tur") == "parca":
                    parca.append(sf.read(io.BytesIO(base64.b64decode(p["wav"])), dtype="float32")[0])
        if not parca:
            return None
        return oz.dinle(np.concatenate(parca), ORAN, hedef)[0]
    except (OSError, ValueError):
        return None


def sabit(kok: Path = KAYIT) -> None:
    """Bir ürünün sabit cümleleri (bugün kullanici/on_kayit/cbs_portal_urun.tsv) Damla ve Kuzey için öğretmenle zaten
    okundu ve denetimden geçti
    (Kaggle seslen-yeniden-uretim-8, sb###): kayda alınır."""
    import librosa
    import soundfile as sf
    kok.mkdir(parents=True, exist_ok=True)
    bitti = {json.loads(x)["anahtar"] for x in (kok / "dizin.jsonl").read_text(encoding="utf-8").splitlines()
             if x} if (kok / "dizin.jsonl").exists() else set()
    say = 0
    for ses in ("damla", "kuzey"):
        d = KOK / "cikti" / "kaggle" / "indirilen" / "seslen-yeniden-uretim-8" / "veri" / ses
        for r in map(json.loads, (d / "olcum.jsonl").read_text(encoding="utf-8").splitlines()):
            if not r["kimlik"].startswith("sb") or not r["gecti"] or r["denetim"]:
                continue
            a = anahtar(ses, r["metin"])
            if a in bitti:
                continue
            x, sr = sf.read(d / "wavs" / f"{r['kimlik']}.flac", dtype="float32")
            (kok / ses).mkdir(exist_ok=True)
            sf.write(kok / ses / f"{a}.flac", librosa.resample(x, orig_sr=sr, target_sr=ORAN), ORAN, subtype="PCM_16")
            _dizine_yaz(kok, {"anahtar": a, "ses": ses, "ton": r["duygu"], "cumle": r["metin"], "durum": "tamam",
                              "dosya": f"{ses}/{a}.flac", "kaynak": r["kimlik"]})
            say += 1
    print(f"{say} hazır öğretmen kaydı alındı")


def durum(kok: Path = KAYIT) -> None:
    dizin = [json.loads(x) for x in (kok / "dizin.jsonl").read_text(encoding="utf-8").splitlines() if x] \
        if (kok / "dizin.jsonl").exists() else []
    sira = {json.loads(x)["anahtar"] for x in (kok / "sira.jsonl").read_text(encoding="utf-8").splitlines() if x} \
        if (kok / "sira.jsonl").exists() else set()
    tamam = sum(r["durum"] == "tamam" for r in dizin)
    print(f"kayıtta {tamam} cümle, geçmeyen {len(dizin) - tamam}, sırada bekleyen {len(sira - {r['anahtar'] for r in dizin})}")


def dinle(kok: Path = KAYIT) -> Path:
    """Kayıttaki sesleri dinlemek için kok/dinle.html (ses başına, en yeni üstte; geçmeyenler ayrı listede)."""
    import html
    dizin = [json.loads(x) for x in (kok / "dizin.jsonl").read_text(encoding="utf-8").splitlines() if x] \
        if (kok / "dizin.jsonl").exists() else []
    sesler: dict[str, list] = {}
    for r in reversed(dizin):
        sesler.setdefault(r["ses"], []).append(r)
    govde = []
    for ses, kayitlar in sesler.items():
        tamam = [r for r in kayitlar if r["durum"] == "tamam"]
        govde.append(f"<details open><summary>{ses} — {len(tamam)} kayıt, {len(kayitlar) - len(tamam)} geçmeyen"
                     "</summary><table>")
        for r in tamam:
            govde.append(f"<tr><td>{r['ton']}</td><td>{html.escape(r['cumle'])}</td><td><audio controls preload=none "
                         f"src=\"{r['dosya']}\"></audio></td></tr>")
        govde.append("</table>")
        gecmeyen = [html.escape(r["cumle"]) for r in kayitlar if r["durum"] != "tamam"]
        if gecmeyen:
            govde.append("<p class=g>Geçmeyen (öğrenci okur): " + " · ".join(gecmeyen) + "</p>")
        govde.append("</details>")
    yol = kok / "dinle.html"
    yol.write_text("<!doctype html><meta charset=utf-8><title>Öğretmen kaydı</title><style>body{font:15px sans-serif;"
                   "margin:16px}td{padding:2px 8px;border-bottom:1px solid #ddd}summary{font-size:18px;margin:12px 0}"
                   ".g{color:#a33}</style><h1>Öğretmen kaydı</h1>" + "".join(govde), encoding="utf-8")
    print(yol)
    return yol


if __name__ == "__main__":
    komut = sys.argv[1] if len(sys.argv) > 1 else "durum"
    {"isci": lambda: isci("--bir" in sys.argv), "sabit": sabit, "durum": durum, "dinle": dinle}[komut]()
