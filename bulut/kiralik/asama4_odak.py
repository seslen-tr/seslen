# -*- coding: utf-8 -*-
"""
ODAK İNCE AYARI (07.10) — tonOlc: Serdar neşeli/kızgın/uyarı ve Damla uyarı/merak nötrden zor ayrılıyor; stil
vektörüyle zorlamak doğallığı bozdu (tonDuzelt.py, git geçmişinde). Model o ses × duyguları daha çok görsün diye
3. aşamanın son ağırlığından 1 kısa tur:
  liste    veri/egitim_vurgu/train_list.txt'ten ağırlıklı liste → veri/egitim_vurgu_odak/
           (ODAK katsayıları; Serdar'ın ve Damla'nın kalanı + öteki seslerden PROVA kayıt unutmasın diye)
  ac       makine (4090 / L40S, Secure)
  kur      StyleTTS2 + paketler (asama3_yeni.KURULUM) + paket_egit + odak.sh
  yukle    listedeki wav'lar, liste, t3-epoch_2nd_00002.pth, Türkçe PL-BERT
  baslat   odak.sh (izleme: runpod.py egit-izle → cikti/ogrenci/t4-epoch_2nd_00000.pth, sonra makine silinir)

    python bulut/kiralik/asama4_odak.py hepsi      (ya da tek adım)
"""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod as r  # noqa: E402
from asama3_yeni import KURULUM, gonder  # noqa: E402

KOK = r.KOK
ODAK = {("serdar", "neseli"): 3, ("serdar", "kizgin"): 3, ("serdar", "uyari"): 3,
        ("damla", "uyari"): 2, ("damla", "merak"): 2}
SES_KALAN = {"serdar": None, "damla": 600}      # None: hepsi bir kez
PROVA = 5000
AKIS = 6                                        # paralel SSH yükleme akışı
HEDEF = KOK / "veri" / "egitim_vurgu_odak"
KAYIT_AD = "Turkce4"                            # makinedeki çıktı klasörü → yerelde önek (Turkce4 → "t4-")
BASLANGIC = KOK / "cikti" / "ogrenci" / "t3-epoch_2nd_00002.pth"   # makinede Turkce3/epoch_2nd_00002.pth adıyla durur


def pod() -> dict:
    return next(x for x in r.podlar() if x.get("rol") == "egitim")


def liste() -> None:
    ozet = json.loads((KOK / "veri" / "egitim" / "ozet.json").read_text(encoding="utf-8"))["duygu"]
    kaynak = KOK / "veri" / "egitim_vurgu"
    satirlar = [s for s in (kaynak / "train_list.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
    rnd = random.Random(5)
    odak, kalan, prova = [], {s: [] for s in SES_KALAN}, []
    for s in satirlar:
        anahtar = s.split("|", 1)[0].removesuffix(".flac")
        ses, duygu = anahtar.split("/")[0], ozet.get(anahtar, "notr")
        if (ses, duygu) in ODAK:
            odak += [s] * ODAK[(ses, duygu)]
        elif ses in SES_KALAN:
            kalan[ses].append(s)
        else:
            prova.append(s)
    son = list(odak)
    for ses, n in SES_KALAN.items():
        son += kalan[ses] if n is None else rnd.sample(kalan[ses], min(n, len(kalan[ses])))
    son += rnd.sample(prova, PROVA)
    rnd.shuffle(son)
    HEDEF.mkdir(parents=True, exist_ok=True)
    (HEDEF / "train_list.txt").write_text("\n".join(son) + "\n", encoding="utf-8")
    for ad in ("val_list.txt", "OOD_texts.txt"):
        (HEDEF / ad).write_text((kaynak / ad).read_text(encoding="utf-8"), encoding="utf-8")
    tek = {s.split("|", 1)[0] for s in son} | {s.split("|", 1)[0] for s in
                                              (kaynak / "val_list.txt").read_text(encoding="utf-8").splitlines() if s}
    (HEDEF / "wavlar.txt").write_text("\n".join(sorted(tek)) + "\n", encoding="utf-8")
    boyut = sum((KOK / "veri" / "egitim" / "wavs" / w).stat().st_size for w in tek)
    print(f"liste: {len(son)} satır (odak {len(odak)}), {len(tek)} ayrı kayıt, {boyut / 1e9:.2f} GB", flush=True)


def ac() -> None:
    r.GPULAR[:] = ["NVIDIA GeForce RTX 4090", "NVIDIA L40S"]       # kalıptaki torch 2.4 5090'da çalışmıyor
    r.ac(1, guvenli=True, rol="egitim")


def kur() -> None:
    p = pod()
    o = r.ssh(p, KURULUM, timeout=3600)
    if "KURULDU" not in o.stdout:
        raise SystemExit("kurulum hatası: " + o.stderr[-800:])
    b = r.BURASI
    gonder([(f, f.name) for f in (b / "egitim").glob("*.py")], "/workspace/paket_egit")
    kur = KOK / "cikti" / "kur.sh"                     # makine yeniden başlarsa devam.sh kurulumu bununla tazeler
    kur.write_text(KURULUM, encoding="utf-8", newline="\n")
    betik = ("odak.sh", "nobetci.sh", "devam.sh")
    gonder([(b / "makine" / f, f) for f in betik] + [(kur, "kur.sh")], "/workspace")
    r.ssh(p, "sed -i 's/\\r$//' " + " ".join(f"/workspace/{f}" for f in (*betik, "kur.sh")), timeout=60)
    print("kuruldu", flush=True)


def yukle() -> None:
    p = pod()
    wavs = KOK / "veri" / "egitim" / "wavs"
    tek = [w for w in (HEDEF / "wavlar.txt").read_text(encoding="utf-8").splitlines() if w]
    o = r.ssh(p, "mkdir -p /workspace/egitim/wavs && cd /workspace/egitim/wavs && find . -type f -printf '%s %P\\n'",
              timeout=300)
    uzak = {yol: int(boy) for boy, _, yol in (s.partition(" ") for s in o.stdout.splitlines())}
    eksik = [w for w in tek if uzak.get(w) != (wavs / w).stat().st_size]      # kaldığı yerden
    print(f"{len(tek)} wav, makinede {len(tek) - len(eksik)}; gönderilecek {len(eksik)}", flush=True)
    t0 = time.time()
    # 17.10: yavaş ağda tek SSH akışı ~80 KB/sn kaldı (TCP penceresi), 6 akış ~3 MB/sn. Başlangıç ağırlığı ayrı akış.
    isler = [([(wavs / w, w) for w in eksik[i::AKIS]], "/workspace/egitim/wavs") for i in range(AKIS)]
    isler.append(([(BASLANGIC, "epoch_2nd_00002.pth")], "/workspace/StyleTTS2/Models/Turkce3"))
    with ThreadPoolExecutor(len(isler)) as ex:
        list(ex.map(lambda a: gonder(*a), [i for i in isler if i[0]]))
    print(f"wav'lar ve başlangıç ağırlığı gönderildi ({(time.time() - t0) / 60:.1f} dk)", flush=True)
    gonder([(HEDEF / ad, ad) for ad in ("train_list.txt", "val_list.txt", "OOD_texts.txt")], "/workspace/egitim_vurgu_odak")
    ara = KOK / "cikti" / "ogrenci" / f"{KAYIT_AD}-ara.pth"       # başka yerde (Modal) yarım kalmış tur: odak.sh oradan sürer
    if ara.exists():
        gonder([(ara, "ara.pth")], f"/workspace/StyleTTS2/Models/{KAYIT_AD}")
        print(f"ara kayıt gönderildi: {ara.name}", flush=True)
    __import__("plbert").plbert_egitime_gonder()
    o = r.ssh(p, "find /workspace/egitim/wavs -name '*.flac' | wc -l; wc -l /workspace/egitim_vurgu_odak/*.txt; "
                 "ls -la /workspace/StyleTTS2/Models/Turkce3 /workspace/StyleTTS2/Utils/PLBERT_tr", timeout=120)
    print(o.stdout, flush=True)


def baslat() -> None:
    p = pod()
    r.ssh(p, "pkill -f '[t]rain_finetune'; true", timeout=60)
    o = r.ssh(p, f"cd /workspace && (GRACE={r.GRACE_DK} KAYIT_AD={KAYIT_AD} setsid nohup bash odak.sh < /dev/null > /dev/null 2>&1 &)",
              timeout=120)
    print("odak ince ayarı başladı" + ("" if o.returncode == 0 else f" HATA {o.stderr[-300:]}"), flush=True)


if __name__ == "__main__":
    adim = sys.argv[1] if len(sys.argv) > 1 else "hepsi"
    for ad, f in (("liste", liste), ("ac", ac), ("kur", kur), ("yukle", yukle), ("baslat", baslat)):
        if adim in (ad, "hepsi"):
            print(f"== {time.strftime('%H:%M')} {ad}", flush=True)
            f()
