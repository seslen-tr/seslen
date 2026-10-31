# -*- coding: utf-8 -*-
"""
ÖĞRENCİ VERİSİ — öğretmen kayıtlarını StyleTTS2 eğitim biçimine çevirir.

Girdi: cikti/veri/<ses>/olcum.jsonl + wavs/ (Kaggle, yerel, RunPod).
Seçim: ölçü kapısından geçen (gecti) + sözcük güveni (araclar/olcu/kelimeGuven.py)
ölçülmüşse ≥ GUVEN_ESIK + 0,6-14 sn (1 sn'den kısa olanlar sessizlikle 1 sn'ye tamamlanır).
Her kayıt:
  - 24 kHz tek kanal
  - kenar sessizliği kırpılır (top_db 40, 0,1 sn pay bırakılır)
  - ses yüksekliği EŞİTLENİR: -20 LUFS, tepe ≤ -1 dBFS. Öğretmen çıktısı sesten
    sese 20 dB oynuyordu (Hasan -12, Zeynep -32 dBFS RMS; kullanıcı "sesler bir
    tık az" dedi, 01.10.2026) — öğrenci hepsini aynı yükseklikte öğrensin.
  - FLAC (24 kHz, 16 bit): yükleme boyutu yarıya iner; StyleTTS2 sf.read ile okur
Metin ogrenci/g2p.py ile IPA'ya çevrilir.
Çıktı (veri/egitim/, gitignore):
  wavs/<ses>/<kimlik>.flac
  train_list.txt, val_list.txt   "<ses>/<kimlik>.flac|<ipa>|<konuşmacı no>"
  OOD_texts.txt                  eğitimde olmayan derlem cümleleri (SLM çekişmeli adım)
  konusmacilar.json, ozet.json   no ↔ ses, ses × duygu dakikaları, kayıt → duygu

    python ogrenci/veri/veriHazirla.py [--sesler damla,serdar,…] [--is 8]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "ogrenci"))
sys.path.insert(0, str(KOK / "ogretmen"))

HEDEF = KOK / "veri" / "egitim"
ORAN = 24000
LUFS = -20.0
TEPE = 10 ** (-1 / 20)
GUVEN_ESIK = 0.3            # kelimeGuven: bu altı "belirsiz okunmuş sözcük" (kalibrasyon: Pıtırcık)
VAL_PAY = 0.01


def kaynak(ses: str, kimlik: str) -> Path:
    for u in (".flac", ".wav"):
        p = KOK / "cikti" / "veri" / ses / "wavs" / f"{kimlik}{u}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{ses}/{kimlik}")


def isle(is_: tuple[str, str]) -> tuple[str, str, float] | None:
    import librosa
    import numpy as np
    import pyloudnorm
    import soundfile as sf
    import soxr
    ses, kimlik = is_
    x, sr = sf.read(kaynak(ses, kimlik), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != ORAN:
        x = soxr.resample(x, sr, ORAN).astype(np.float32)
    _, (a, b) = librosa.effects.trim(x, top_db=40)
    pay = int(0.1 * ORAN)
    x = x[max(0, a - pay): b + pay]
    if len(x) < 0.6 * ORAN:        # çok kısa: StyleTTS2 kırpma / hizalama için riskli (01.10)
        return None
    if len(x) < 1.0 * ORAN:        # kısa cümle ("Evet.", "Dikkat!"): iki yana sessizlikle 1 sn'ye (10.10)
        eksik = int(1.0 * ORAN) - len(x)
        x = np.pad(x, (eksik // 2, eksik - eksik // 2))
    yuk = pyloudnorm.Meter(ORAN).integrated_loudness(x)
    if not np.isfinite(yuk):
        return None
    x = x * 10 ** ((LUFS - yuk) / 20)
    tepe = np.abs(x).max()
    if tepe > TEPE:
        x = x * (TEPE / tepe)
    hedef = HEDEF / "wavs" / ses / f"{kimlik}.flac"
    hedef.parent.mkdir(parents=True, exist_ok=True)
    sf.write(hedef, x, ORAN, subtype="PCM_16")
    return ses, kimlik, len(x) / ORAN


def main() -> int:
    from g2p import g2p
    from kadro import secilenler
    ap = argparse.ArgumentParser()
    ap.add_argument("--sesler", default="")
    ap.add_argument("--is", dest="is_", type=int, default=8)
    a = ap.parse_args()
    sesler = [s for s in a.sesler.split(",") if s] or list(secilenler())
    konusmaci = {s: i for i, s in enumerate(sesler)}

    secim: dict[tuple[str, str], dict] = {}
    for ses in sesler:
        for s in (KOK / "cikti" / "veri" / ses / "olcum.jsonl").read_text(encoding="utf-8").splitlines():
            o = json.loads(s)
            # Fısıltıda netlik (harmonik güç) doğası gereği düşük: yalnız doğruluk kapısı (03.10: netlik
            # yüzünden fısıltıların %51'i kapıda kalıyordu, Whisper doğruluğu ortalama 0,97 iken)
            gecti = o["gecti"] or (o.get("duygu") == "fisilti" and o["dogruluk"] >= 0.90)
            if not gecti or not (0.6 <= o["sn"] <= 14):
                continue
            if o.get("min_guven", 1.0) < GUVEN_ESIK:
                continue
            secim[(ses, o["kimlik"])] = o          # aynı kimlik iki kez varsa sonuncusu
    print(f"{len(secim)} kayıt işlenecek ({len(sesler)} ses)", flush=True)

    sureler: dict[tuple[str, str], float] = {}
    with ProcessPoolExecutor(a.is_) as ex:
        for i, r in enumerate(ex.map(isle, secim, chunksize=64), 1):
            if r:
                sureler[(r[0], r[1])] = r[2]
            if i % 2000 == 0:
                print(f"  {i}/{len(secim)}", flush=True)

    satirlar, ozet, duygu = [], {}, {}
    for (ses, kimlik), sn in sureler.items():
        o = secim[(ses, kimlik)]
        d = o.get("duygu", "notr")
        satirlar.append((ses, f"{ses}/{kimlik}.flac|{g2p(o['metin'])}|{konusmaci[ses]}"))
        ozet.setdefault(ses, {}).setdefault(d, 0.0)
        ozet[ses][d] += sn / 60
        duygu[f"{ses}/{kimlik}"] = d
    rnd = random.Random(7)
    rnd.shuffle(satirlar)
    val, train = [], []
    for ses in sesler:
        s = [x for k, x in satirlar if k == ses]
        n = max(4, int(len(s) * VAL_PAY))
        val += s[:n]
        train += s[n:]
    rnd.shuffle(train)
    (HEDEF / "train_list.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (HEDEF / "val_list.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    # OOD: eğitimde okunmamış derlem cümleleri (SLM çekişmeli adımda "görmediği metin")
    okunan = {secim[k]["metin"] for k in sureler}
    korpus = [json.loads(s)["metin"] for s in (KOK / "veri" / "korpus.jsonl").read_text(encoding="utf-8").splitlines()]
    ood = [g2p(m) for m in korpus if m not in okunan and len(m) > 25]
    (HEDEF / "OOD_texts.txt").write_text("\n".join(ood) + "\n", encoding="utf-8")

    (HEDEF / "konusmacilar.json").write_text(json.dumps(konusmaci, ensure_ascii=False, indent=1), encoding="utf-8")
    (HEDEF / "ozet.json").write_text(json.dumps({"dakika": {s: {d: round(v, 1) for d, v in x.items()} for s, x in ozet.items()},
                                                 "duygu": duygu}, ensure_ascii=False), encoding="utf-8")
    toplam = sum(sureler.values()) / 3600
    print(f"train {len(train)}, val {len(val)}, OOD {len(ood)}; toplam {toplam:.1f} sa")
    for ses in sesler:
        print(f"  {ses:9s} " + " ".join(f"{d}:{v:.0f}" for d, v in sorted(ozet.get(ses, {}).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
