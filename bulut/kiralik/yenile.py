# -*- coding: utf-8 -*-
"""
VERİ YENİLEME + İNCE AYAR (10.10) — dinlemede yakalanan üç kalıcı sorun için:
  1. Fatma nineye benzemiyordu (öğretmen kaydının perdesi 323 Hz, genç kız): yeni tasarım fatma-5 (231 Hz, hırıltılı)
     seçildi; Fatma'nın bütün verisi bu sesle yeniden okunur. Hırıltı netlik ölçüsünü düşürür (17-18): kapı 15.
  2. Kısa cümleler bozuluyordu ("Sınavı geçtim!" → "sınava"): verinin %0,3'ü 1-2 sözcüktü. veri/kisa_cumleler.tsv
     (245 cümle) açık sesler + Damla, Kuzey, Serdar tarafından okunur; masal dili için veri/masal_cumleler.tsv.
  3. Vurgu (turkce-vurgu 0.1.1: VARmış, acele ETme): bütün listeler yeniden okunuşlanır (ogrenci/veri/vurguListe.py).
Ardından 3. aşama ağırlığından (t3) bir tur ince ayar → t6 (asama4_odak'ın makine adımları).
  4. (11.10, t7) Hasan'ın telaffuzu iki modelde de sorunluydu, öğretmen perdesi 246 Hz: yeni tasarım hasan-6 (165 Hz)
     ile verisi baştan okunur; ince ayar t6'nın üzerine (Fatma ve kısa cümle kazanımları korunsun).
Turlar TURLAR'da; TUR hangisinin çalışacağını seçer.

Adımlar (her biri ayrı çağrılabilir):
  plan       veri/plan_yeniden/plan1.jsonl (Fatma: bütün cümleler + kısa + masal; ref2 Kaggle'da üretilir),
             plan2.jsonl (öteki seslerin kısa + masal cümleleri), refler.json, ayar.json
             → python bulut/kaggle/kaggleIs.py yeniden | yeniden-durum | yeniden-indir
  birlestir  indirilenler cikti/veri/'ye: Fatma'nın eskisi cikti/veri/fatma_eski/, yenisi yerine; ötekilere eklenir
  veri       ogrenci/veri/veriHazirla.py (bütün sesler) + ogrenci/veri/vurguListe.py
  liste      ağırlıklı liste → veri/egitim_vurgu_<tur>/ (yeni ses ×2, kısa ×3, masal ×2, kalandan PROVA)
  egit       makine aç, kur, yükle, başlat (asama4_odak), izle → cikti/ogrenci/<tur>-epoch_2nd_00000.pth

    python bulut/kiralik/yenile.py plan
"""
from __future__ import annotations

import functools
import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod as r  # noqa: E402

KOK = r.KOK
PLAN = KOK / "veri" / "plan_yeniden"
# yeni: verisi baştan okunan ses → seçilen aday; kisa: kısa/masal cümleleri bu turda okunacak öteki sesler
TURLAR = {
    "t6": {"yeni": {"fatma": "fatma-5"}, "kisa": ["ayse", "zeynep", "emre", "hasan", "koca", "damla", "kuzey", "serdar"],
           "kapi_netlik": {"fatma": 15.0}, "baslangic": "t3-epoch_2nd_00002.pth", "prova": 5000},
    "t7": {"yeni": {"hasan": "hasan-6"}, "kisa": [], "kapi_netlik": {}, "baslangic": "t6-epoch_2nd_00000.pth",
           "prova": 4000},
    # 14.10 pilot Türkçe ön eğitim: YODAS2 temiz konuşma (onEgitim.py, git geçmişinde) + bizim 16 sesten prova, t7'den
    "t8": {"yeni": {}, "kisa": [], "kapi_netlik": {}, "baslangic": "t7-epoch_2nd_00000.pth", "prova": 8000,
           "ek": KOK / "veri" / "egitim_vurgu_on" / "yodas.txt"},
}
# 15.10 seçim sayfası (araclar/deney/yolKarsilastir.py): 11 ses "1. yol" (Chatterbox Türkçesi + VoxCPM2 tarif), Can
# yeni erkek çocuk (kaggleIs CAN, aday A). Verileri baştan D kipiyle (bulut/kaggle/yeniden.py) okunur; Damla, Elif,
# Kara, Kuzey eski verileriyle kalır (Kuzey'in uzay sesi eğitimden sonra sesUzayi.py ile, git geçmişinde). 22 sa YODAS2 ön eğitimle.
_YOL1 = {s: (f"cikti/kaggle/dene/ref/iki_{s}.wav", s) for s in
         ("ayse", "serdar", "zeynep", "emre", "selin", "fatma", "ece", "pitircik", "koca", "firfir")}
_YOL1 |= {"hasan": ("cikti/kaggle/dene/ref/iki_serdar.wav", "YASLI"), "can": ("cikti/kaggle/dene/ref/iki_emre.wav", "OGLAN")}
TURLAR["t9"] = {"yeni": {s: f"{s}-yol1" for s in _YOL1}, "yol1": _YOL1, "kisa": [], "kapi_netlik": {"fatma": 15.0},
                "baslangic": "t8-epoch_2nd_00000.pth", "prova": 4000, "adet": 900, "parca": 4,
                # 16.10: ilk iki parça (900 cümle) bitti; Kaggle kotası doldu → kalan sesler 700 cümle, parça 3-4;
                # ilk 6 sesin denetime takılan kayıtları parça 5'te yeniden okunur (araclar/olcu/denetim.py)
                "tamam": ["ayse", "emre", "koca", "fatma", "firfir", "selin"], "adet_kalan": 700, "ilk_parca": 3,
                "parca_kalan": 2,
                "ek": KOK / "veri" / "egitim_vurgu_on" / "yodas.txt", "ek_adet": 8000}   # ~18 bin adım: 4090da ~12 sa
# 19-20.10 dinleme: t9'da bütün seslerde 1. yol seçildi (Elif şimdiki). Damla, Kuzey, Kara 1. yol sesine geçer; Can
# daha ince G sesine (kaggleIs.OGLAN8). Damla ve Kuzey ürünün sabit cümlelerini de okur (31.10'dan beri kullanici/on_kayit/cbs_portal_urun.tsv): eğitim
# verisi + sunucu önbelleği için öğretmen kaydı. Kaggle parça 6-7; birleştirme yalnız bu turun parçalarından.
TURLAR["t10"] = {"yeni": {s: f"{s}-yol1" for s in ("damla", "kuzey", "kara", "can")},
                 "yol1": {"damla": ("cikti/kaggle/dene/ref/iki_damla.wav", "damla"),
                          "kuzey": ("cikti/kaggle/dene/ref/iki_kuzey.wav", "kuzey"),
                          "kara": ("cikti/kaggle/dene/ref/iki_kara.wav", "kara"),
                          "can": ("cikti/kaggle/dene/ref/can_a_p3.wav", "OGLAN8")},
                 "sabit": ["damla", "kuzey"], "kisa": [], "kapi_netlik": {"can": 15.0}, "adet": 900, "parca": 2, "ilk_parca": 6,  # RunPod: 3 parça
                 "parcalar": [8, 9], "baslangic": "t9-epoch_2nd_00000.pth", "prova": 2000,   # Modal kalan kredisi: ~5.500 adım
                 "ek": KOK / "veri" / "egitim_vurgu_on" / "yodas.txt", "ek_adet": 1000}
TUR = "t10"
YENI_SES, KISA_SESLER = TURLAR[TUR]["yeni"], TURLAR[TUR]["kisa"]
MASAL_SESLER = ["fatma", "hasan", "ayse"]
KAPI_NETLIK = TURLAR[TUR]["kapi_netlik"]
CARPAN = {"yeni": 2, "ks": 3, "ms": 2}               # liste: yeni sesin kaydı, kısa, masal
PROVA = TURLAR[TUR]["prova"]
HEDEF = KOK / "veri" / f"egitim_vurgu_{TUR}"
KAYIT_AD = "Turkce" + TUR[1:]
INDIRILEN = KOK / "cikti" / "kaggle" / "indirilen"


def _kadro():
    sys.path.insert(0, str(KOK / "ogretmen"))
    import kadro
    return kadro


def _tsv(ad: str, onek: str) -> list[dict]:
    satir = [s.split("\t") for s in (KOK / "veri" / ad).read_text(encoding="utf-8").splitlines()
             if s.strip() and not s.startswith("#")]
    return [{"kimlik": f"{onek}{i:03d}", "duygu": d, "metin": m} for i, (d, m) in enumerate(satir, 1)]


def _kip(ses: str, duygu: str) -> str:
    return _kadro().KIP_NOTR.get(ses, "B") if duygu == "notr" else "B"


def _tarif(ad: str) -> str:
    """Karakter tarifi (bulut/kaggle/kaggleIs.py): sesin adıysa TUM_TARIF, yoksa adı geçen sabit (YASLI, OGLAN)."""
    sys.path.insert(0, str(KOK / "bulut" / "kaggle"))
    import kaggleIs as k
    return k.TUM_TARIF[ad] if ad in k.TUM_TARIF else getattr(k, ad)


_EYLEM_IPUCU = {"yasak": "firmly telling the listener not to do it", "emir": "giving the listener a direct instruction",
                "uyari": "warning the listener directly", "soru": "asking the listener a real question",
                "ozur": "sincerely apologizing to the listener", "rica": "politely asking the listener",
                "oneri": "suggesting something to the listener"}


@functools.lru_cache(maxsize=1)
def _adlar() -> set[str]:
    import turkce_vurgu
    return set((Path(turkce_vurgu.__file__).parent / "veri" / "adlar.txt").read_text(encoding="utf-8").split())


def _ipucu(kimlik: str, metin: str) -> str:
    """Konuşma eylemi (turkce-duygu) ve hitap: öğretmen "Kaan, ... kesme." cümlesini düz okuyordu (16.10)."""
    from turkce_duygu import cozumle
    parca = []
    m = re.match(r"^([A-ZÇĞİÖŞÜ][a-zçğıöşü]+),", metin)
    if m and m.group(1).replace("İ", "i").lower() in _adlar():
        parca.append(f"speaking directly to {m.group(1)}, calling them by name")
    e = _EYLEM_IPUCU.get(cozumle(metin).eylem)
    if e:
        parca.append(e)
    if kimlik.startswith("ms"):
        parca.append("telling a fairy tale slowly and warmly")
    return ", ".join(parca)


def _denetim_takilan(ses: str) -> set[str]:
    takilan = set()
    for d in INDIRILEN.glob(f"seslen-yeniden-uretim-*/veri/{ses}/denetim.jsonl"):
        takilan |= {r["kimlik"] for r in map(json.loads, d.read_text(encoding="utf-8").splitlines()) if r["bayrak"]}
    return takilan


def _sabit(ses: str) -> list[dict]:
    """Ürünün sabit cümleleri (kullanici/on_kayit/cbs_portal_urun.tsv), {ad} sesin adıyla: kimlik sb001…"""
    satir = [x.split("	", 1) for x in (KOK / "kullanici" / "on_kayit" / "cbs_portal_urun.tsv").read_text(encoding="utf-8").splitlines()
             if x.strip() and not x.startswith("#")]
    return [{"kimlik": f"sb{i:03d}", "duygu": d, "metin": m.replace("{ad}", ses.capitalize())}
            for i, (d, m) in enumerate(satir, 1)]


def plan_yol1() -> None:
    """D kipi: her ses plandan ADET cümle (duygular karışık) + kısa + masal; parçalara dengeli dağıtılır. "tamam"daki
    seslerden yalnız denetime takılanlar (ayrı son parça); kalanlar "adet_kalan" cümleyle."""
    tur = TURLAR[TUR]
    kisa, masal = _tsv("kisa_cumleler.tsv", "ks"), _tsv("masal_cumleler.tsv", "ms")
    rnd = random.Random(9)
    gruplar, onar, refler, tarif = [], [], {}, {}
    tamam = set(tur.get("tamam", []))
    for ses, (ref, tarif_ad) in tur["yol1"].items():
        tum = {}
        for f in ("plan", "plan_yeni"):
            yol = KOK / "veri" / f / f"{ses}.jsonl"
            for x in (yol.read_text(encoding="utf-8").splitlines() if yol.exists() else []):
                if x.strip():
                    c = json.loads(x)
                    tum[c["kimlik"]] = {k: c[k] for k in ("kimlik", "metin", "duygu")}
        secim = rnd.sample(sorted(tum.values(), key=lambda c: c["kimlik"]), min(tur["adet"], len(tum)))
        refler[ses], tarif[ses] = [ref, None], _tarif(tarif_ad)
        if ses in tamam:
            takilan = _denetim_takilan(ses)
            onar += [{**c, "ses": ses, "kip": "D", "ipucu": _ipucu(c["kimlik"], c["metin"])}
                     for c in secim + kisa + (masal if ses in MASAL_SESLER else []) if c["kimlik"] in takilan]
            continue
        secim = secim[: tur.get("adet_kalan", tur["adet"])]
        sabit = _sabit(ses) if ses in tur.get("sabit", []) else []
        gruplar.append([{**c, "ses": ses, "kip": "D", "ipucu": _ipucu(c["kimlik"], c["metin"])}
                        for c in secim + kisa + (masal if ses in MASAL_SESLER else []) + sabit])
    parcalar: list[list[dict]] = [[] for _ in range(tur.get("parca_kalan", tur["parca"]))]
    for g in sorted(gruplar, key=len, reverse=True):
        min(parcalar, key=len).extend(g)
    if onar:
        parcalar.append(onar)
    for i, p in enumerate(parcalar, tur.get("ilk_parca", 1)):
        (PLAN / f"plan{i}.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in p), encoding="utf-8")
        print(f"plan{i}: {len(p)} cümle ({', '.join(sorted({c['ses'] for c in p}))})", flush=True)
    (PLAN / "refler.json").write_text(json.dumps(refler, ensure_ascii=False, indent=1), encoding="utf-8")
    (PLAN / "ayar.json").write_text(json.dumps({"aday": 2, "sure_dk": 690, "kapi_netlik": KAPI_NETLIK, "tarif": tarif},
                                               ensure_ascii=False), encoding="utf-8")


def plan() -> None:
    kadro = _kadro()
    kisa, masal = _tsv("kisa_cumleler.tsv", "ks"), _tsv("masal_cumleler.tsv", "ms")
    PLAN.mkdir(parents=True, exist_ok=True)
    for eski in PLAN.glob("plan*.jsonl"):
        eski.unlink()
    if TURLAR[TUR].get("yol1"):
        return plan_yol1()
    parcalar, refler = [], {}
    for ses, aday in YENI_SES.items():
        tum = {}
        for f in ("plan", "plan_yeni"):
            yol = KOK / "veri" / f / f"{ses}.jsonl"
            for x in (yol.read_text(encoding="utf-8").splitlines() if yol.exists() else []):
                if x.strip():
                    c = json.loads(x)
                    tum[c["kimlik"]] = {k: c[k] for k in ("kimlik", "metin", "duygu", "kip") if k in c}
        cumleler = list(tum.values()) + kisa + (masal if ses in MASAL_SESLER else [])
        parcalar.append([{**c, "ses": ses, "kip": c.get("kip") or _kip(ses, c["duygu"])} for c in cumleler])
        refler[ses] = [f"cikti/adaylar/{aday}/ref.wav", None]           # ref2 Kaggle'da üretilir
    diger = []
    for ses in KISA_SESLER:
        aday = kadro.secilenler()[ses]
        ref = f"cikti/adaylar/{aday}/ref.wav"
        refler[ses] = [ref if (KOK / ref).exists() else f"cikti/tazele/{aday}/ref.wav", f"cikti/tazele/{aday}/ref2.wav"]
        diger += [{**c, "ses": ses, "kip": _kip(ses, c["duygu"])} for c in kisa + (masal if ses in MASAL_SESLER else [])]
    if diger:
        parcalar.append(diger)
    for i, p in enumerate(parcalar, 1):
        (PLAN / f"plan{i}.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in p), encoding="utf-8")
        print(f"plan{i}: {len(p)} cümle ({', '.join(sorted({c['ses'] for c in p}))})", flush=True)
    (PLAN / "refler.json").write_text(json.dumps(refler, ensure_ascii=False, indent=1), encoding="utf-8")
    (PLAN / "ayar.json").write_text(json.dumps({"aday": 2, "sure_dk": 690, "kapi_netlik": KAPI_NETLIK}), encoding="utf-8")


def birlestir() -> None:
    sys.path.insert(0, str(KOK / "ogretmen"))
    from veriUret import metadata_yaz
    kaynaklar = sorted(INDIRILEN.glob("seslen-yeniden-uretim-*/veri/*"))
    if TURLAR[TUR].get("parcalar"):                    # t10: Can'ın t9 (A) verisi karışmasın
        kaynaklar = [k for k in kaynaklar if int(k.parents[1].name.rsplit("-", 1)[1]) in TURLAR[TUR]["parcalar"]]
    if not kaynaklar:
        raise SystemExit("indirilen yok: önce python bulut/kaggle/kaggleIs.py yeniden-indir")
    for ses, aday in YENI_SES.items():                # yeni ses: eski klasör bir kez yedeklenir, yerine boşu
        yer = KOK / "cikti" / "veri" / ses
        if not (KOK / "cikti" / "veri" / f"{ses}_eski").exists():
            yer.rename(KOK / "cikti" / "veri" / f"{ses}_eski")
        if yer.exists():
            shutil.rmtree(yer)
        (yer / "wavs").mkdir(parents=True)
        (yer / "olcum.jsonl").write_text("", encoding="utf-8")
        t = KOK / "cikti" / "tazele" / aday
        t.mkdir(parents=True, exist_ok=True)
        ref2 = next(r for r in sorted(INDIRILEN.glob(f"seslen-yeniden-uretim-*/ref/{ses}2.wav"), reverse=True)
                    if not TURLAR[TUR].get("parcalar")
                    or int(r.parents[1].name.rsplit("-", 1)[1]) in TURLAR[TUR]["parcalar"])   # t10 Can: A değil G'nin kaydı
        ref = KOK / "cikti" / "adaylar" / aday / "ref.wav"
        shutil.copy(ref if ref.exists() else ref2, t / "ref.wav")      # 1. yol seslerinde tasarım kaydı yok
        shutil.copy(ref2, t / "ref2.wav")
    # Aynı kimlik birden çok parçada olabilir (16.10: denetime takılanlar parça 5'te yeniden okundu): sonraki parça
    # kazanır. Denetimi sonradan yapılmış parçada (denetim.jsonl) takılan kayıt eğitime girmez (gecti=False).
    toplu: dict[str, dict[str, tuple[dict, Path]]] = {}
    for k in sorted(kaynaklar, key=lambda k: int(k.parents[1].name.rsplit("-", 1)[1])):
        takilan = set()
        if (k / "denetim.jsonl").exists():
            takilan = {r["kimlik"] for r in map(json.loads, (k / "denetim.jsonl").read_text(encoding="utf-8").splitlines())
                       if r["bayrak"]}
        for o in (json.loads(x) for x in (k / "olcum.jsonl").read_text(encoding="utf-8").splitlines() if x):
            if o["ses"] in KAPI_NETLIK:        # 25.10: ince çocuk sesi (Can G) netlikte doğal olarak düşük (medyan 16,6)
                o = {**o, "gecti": o["dogruluk"] >= 0.90 and o["netlik"] >= KAPI_NETLIK[o["ses"]] and not o["denetim"]}
            if o["kimlik"] in takilan:
                o = {**o, "gecti": False, "denetim": ["sonradan denetim"]}
            toplu.setdefault(k.name, {})[o["kimlik"]] = (o, k)
    for ses, kayit in toplu.items():
        yer = KOK / "cikti" / "veri" / ses
        eski = {json.loads(x)["kimlik"] for x in (yer / "olcum.jsonl").read_text(encoding="utf-8").splitlines() if x}
        with open(yer / "olcum.jsonl", "a", encoding="utf-8") as f:
            for kim, (o, k) in kayit.items():
                if kim in eski and ses not in YENI_SES:
                    continue
                shutil.copy(k / "wavs" / f"{kim}.flac", yer / "wavs" / f"{kim}.flac")
                f.write(json.dumps({**o, "yer": f"kaggle:{k.parents[1].name}"}, ensure_ascii=False) + "\n")
        gecen = sum(o["gecti"] for o, _ in kayit.values())
        print(f"{ses}: {len(kayit)} kayıt (geçen {gecen}); metadata {metadata_yaz(ses)}", flush=True)


def veri() -> None:
    import os
    py = str(KOK / ".venv" / "Scripts" / "python.exe")
    # 12.10: 8 işçinin her biri çekirdek sayısı kadar BLAS iş parçacığı açınca bellek bitti (boş 2,6 GB) → 3 × 1
    ortam = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for betik, ek in (("veriHazirla.py", ["--is", "3"]), ("vurguListe.py", [])):
        print(f"== {betik}", flush=True)
        subprocess.run([py, str(KOK / "ogrenci" / "veri" / betik), *ek], check=True, cwd=KOK, env=ortam)


def liste() -> None:
    kaynak = KOK / "veri" / "egitim_vurgu"
    satirlar = [s for s in (kaynak / "train_list.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
    rnd = random.Random(6)
    son, prova, say = [], [], {k: 0 for k in CARPAN}
    for s in satirlar:
        ses, kimlik = s.split("|", 1)[0].removesuffix(".flac").split("/")
        bu_tur = ses in YENI_SES or ses in KISA_SESLER
        tur = (kimlik[:2] if kimlik[:2] in ("ks", "ms") else "yeni" if ses in YENI_SES else None) if bu_tur else None
        if tur:
            son += [s] * CARPAN[tur]
            say[tur] += 1
        else:
            prova.append(s)
    son += rnd.sample(prova, min(PROVA, len(prova)))
    ek = TURLAR[TUR].get("ek")
    if ek:                                             # ön eğitim verisi (YODAS2): her segment bir kez
        satir_ek = [x for x in ek.read_text(encoding="utf-8").splitlines() if x.strip()]
        if TURLAR[TUR].get("ek_adet"):                 # t9: süre/ücret sınırı (bulut/modal/egitModal.py hiz)
            satir_ek = rnd.sample(satir_ek, min(TURLAR[TUR]["ek_adet"], len(satir_ek)))
        son += satir_ek
        say["ek"] = len(son) - sum(say.values()) - min(PROVA, len(prova))
    rnd.shuffle(son)
    HEDEF.mkdir(parents=True, exist_ok=True)
    (HEDEF / "train_list.txt").write_text("\n".join(son) + "\n", encoding="utf-8")
    for ad in ("val_list.txt", "OOD_texts.txt"):
        shutil.copy(kaynak / ad, HEDEF / ad)
    tek = {s.split("|", 1)[0] for s in son} | {s.split("|", 1)[0] for s in
                                              (kaynak / "val_list.txt").read_text(encoding="utf-8").splitlines() if s}
    (HEDEF / "wavlar.txt").write_text("\n".join(sorted(tek)) + "\n", encoding="utf-8")
    boyut = sum((KOK / "veri" / "egitim" / "wavs" / w).stat().st_size for w in tek)
    print(f"liste: {len(son)} satır {say} + prova {min(PROVA, len(prova))}; {len(tek)} ayrı kayıt, {boyut / 1e9:.2f} GB",
          flush=True)


def ag_kontrol() -> None:
    """Makine kiralamadan önce: RunPod SSH'ı yüksek bir porttan gelir. 10.10'da bağlanılan ağ yalnız 443'e izin
    veriyordu; iki makine SSH bekleyerek boşa ücret yazdı. Dışarıya yüksek port kapalıysa kiralanmaz."""
    import socket
    for port in (16167, 19425, 22):
        try:
            socket.create_connection(("portquiz.net", port), timeout=8).close()
            return
        except OSError:
            continue
    raise SystemExit("bu ağ dışarıya yalnız 443'e izin veriyor (SSH kapalı): makine kiralanmadı. Başka bir ağ "
                     "(ev interneti, telefon paylaşımı) gerekli.")


def egit() -> None:
    ag_kontrol()
    import asama4_odak as a4
    a4.HEDEF, a4.KAYIT_AD = HEDEF, KAYIT_AD
    a4.BASLANGIC = KOK / "cikti" / "ogrenci" / TURLAR[TUR]["baslangic"]
    for f in (a4.ac, a4.kur, a4.yukle, a4.baslat):
        print(f"== {time.strftime('%H:%M')} egit/{f.__name__}", flush=True)
        f()
    r.egit_izle()


if __name__ == "__main__":
    adim = sys.argv[1] if len(sys.argv) > 1 else ""
    adimlar = (("plan", plan), ("birlestir", birlestir), ("veri", veri), ("liste", liste), ("egit", egit))
    if adim not in {a for a, _ in adimlar}:
        raise SystemExit(__doc__)
    for ad, f in adimlar:
        if adim == ad:
            print(f"== {time.strftime('%H:%M')} {ad}", flush=True)
            f()
