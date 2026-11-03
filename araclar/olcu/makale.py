# -*- coding: utf-8 -*-
"""
MAKALE SINAVI (03.11.2026) — uzun belgeyi (Markdown / HTML / düz metin) seslen-tts paketinin KENDİ yoluyla
seslendirir ve her cümleyi denetler. Sebep: kullanıcı makalelerini seslendirdi, sonuç "rezalet"ti (Markdown işareti
okundu, tire duraklama oldu, kısaltma, model adı ve birim okunmadı, duygu cümle cümle sıçradı, hız 4,6–7,3 hece/sn
oynadı). Her sürüm yayımlanmadan önce bu sınavdan geçer (--esik).

Denetimler (cümle başına):
  kalan     okunuşta okunmadan kalan işaret ya da Türkçede olmayan harf (# * < > | _ / q w x, rakam…)
  ctc       Türkçe wav2vec2 CTC okunuş denetimi (araclar/olcu/denetim.py) takıldı mı
  hiz       hece/sn, sesin doğal hızına (Seslen.dogal_hiz × üslup hızı) göre ±%25 dışında mı (6 heceden kısa sayılmaz)
  ton       belge üslubunda olmaması gereken duygu (heyecan, kızgın, üzgün, fısıltı) seçildi mi
Eşikler (ESIK): kalan 0, belgede yasak ton 0, hız dışı ≤ %5, CTC takılan ≤ %25, çökme yok. CTC eşiği gerilemeyi
yakalar, kusursuzluğu değil: teknik makalede CTC nadir sözcükte de takılıyor (03.11, 6 makale: %20–22; kısa genel
cümlelerde %3, ogrenci/aktar/acikOlc.py).

    python araclar/olcu/makale.py [belge.md …] [--ses zeynep] [--esik] [--cikti klasör]
Belge verilmezse yerleşik ORNEK belge (her zor yapıdan bir örnek). Çıktı: cikti/makale/<belge>/ (cümle wav'ları,
<belge>.wav bütün okuma, rapor.json) ve cikti/makale/dinle.html. seslen-tts deposu: SESLEN_TTS_DEPO ya da
Masaüstü/ses_projeleri/seslen-tts; model: SESLEN_TTS_MODEL ya da cikti/acik/seslen-tts.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[2]
DEPO = Path(os.environ.get("SESLEN_TTS_DEPO", KOK.parents[1] / "ses_projeleri" / "seslen-tts"))
sys.path[:0] = [str(DEPO), str(KOK / "araclar" / "olcu")]
ESIK = {"kalan": 0, "yasak_ton": 0, "hiz_disi": 0.05, "ctc": 0.25}
HIZ_SINIR = (0.75, 1.25)
_KALAN = re.compile(r"[^a-zçğıöşüâîû\s.,;:!?…\"]|[qwx]")

ORNEK = """<!-- Bu yorum okunmaz. -->
# Türkçe LLM'leri ölçtük: 6 model, 3 görev

Hazır İngilizce ölçütler (benchmark) Türkçe kamu işlerinde pek bir şey söylemiyor. Ben de **EVREN**'in LLM
API'sindeki modelleri kendi setimle ölçtüm; sonuçlar %94–100 arasında.

## Sonuçlar (Ekim 2026)

| Model | Doğruluk | Süre |
|---|---|---|
| deepseek-v4.1-flash | **%81,2** | 3,5 sn |
| qwen3.8-flash-next | %77,2 | 2,1 sn |
| gemma-4-31b | %75,2 | 4 sn |

**1. Bağlam verilince fark kapanıyor.** Modele ilgili metni vermek (RAG) daha güvenli.

**2. Cevap sınırına dikkat.** İki model %18–20 oranında boş cevap döndürdü.

- Sentinel-2 uydusundan 10 m çözünürlüklü görüntüler kullanıldı.
- PostgreSQL ve PostGIS üzerinde GeoJSON verisi, 24 kHz ses ve 16-bit örnekler.
- e-Devlet, TÜİK ve TKGM verileriyle Ar-Ge çalışması yapıldı.

Model v0.2.1 sürümünde 1.250,75 TL'lik bütçeyle, 15 Mart 2026 tarihinde, saat 14:30'da yayımlandı. Llama 3.1 ve
Qwen2.5 da karşılaştırıldı. Gerçek zamanın yaklaşık 2,5–5 katı hızlı; LinkedIn + Medium'da paylaşıldı.

Veri setleri herkese açık: https://evren.ssyz.org.tr/datasets/u43bd0650c87e/tr-mevzuat-sc

```python
print("kod okunmaz")
```

#EVREN #LLM #Türkçe
"""


def sinav(s, ad: str, metin: str, ses: str, hedef: Path) -> dict:
    import denetim
    import soundfile as sf
    from seslen_tts import ORAN
    from seslen_tts.model import _UNLU
    from turkce_vurgu import normallestir
    hedef.mkdir(parents=True, exist_ok=True)
    satir, dalga, t0 = [], [], time.perf_counter()
    try:
        parcalar = list(s.seslendir(metin, ses=ses))
    except Exception as h:                                           # çökme sınavı düşürür
        return {"belge": ad, "cokme": f"{type(h).__name__}: {h}", "satir": []}
    from seslen_tts.seslen import _cumleler
    uslup_hizi = s._okuma(" ".join(_cumleler(metin)), "otomatik", None)[1]["hiz"]      # seslendir'in seçtiği
    for i, p in enumerate(parcalar):
        n = normallestir(p.cumle)
        hece = sum(c in _UNLU for c in s.model.okunus(p.cumle))
        hiz = hece / max(1e-6, len(p.ses) / ORAN)
        dogal = s.dogal_hiz(ses, p.ton) * uslup_hizi
        ref = " ".join(re.findall(r"[a-zçğıöşüâîû]+", n))
        d = denetim.denetle(p.ses, ORAN, ref) if ref else {"bayrak": False, "duyulan": ""}
        sf.write(hedef / f"{i:03d}.wav", p.ses, ORAN)
        dalga += [p.ses, np.zeros(int(ORAN * p.sessizlik_ms / 1000), np.float32)]
        satir.append({"cumle": p.cumle, "okunus": n, "ton": p.ton, "belge": p.bilgi.get("belge", False),
                      "kalan": sorted(set(_KALAN.findall(n))), "ctc": bool(d["bayrak"]), "duyulan": d.get("duyulan", ""),
                      "hiz": round(hiz, 2), "hiz_orani": round(hiz / dogal, 2) if hece >= 6 else None})
    sf.write(hedef.parent / f"{ad}.wav", np.concatenate(dalga) if dalga else np.zeros(1, np.float32), ORAN)
    olcu = [r["hiz_orani"] for r in satir if r["hiz_orani"]]
    ozet = {"belge": ad, "cumle": len(satir), "sure_sn": round(sum(len(x) for x in dalga) / ORAN, 1),
            "uretim_sn": round(time.perf_counter() - t0, 1), "kalan": sum(bool(r["kalan"]) for r in satir),
            "yasak_ton": sum(r["belge"] and r["ton"] in ("heyecan", "kizgin", "uzgun", "fisilti") for r in satir),
            "hiz_disi": round(sum(not HIZ_SINIR[0] <= o <= HIZ_SINIR[1] for o in olcu) / max(1, len(olcu)), 3),
            "ctc": round(sum(r["ctc"] for r in satir) / max(1, len(satir)), 3),
            "ton_degisimi": sum(a["ton"] != b["ton"] for a, b in zip(satir, satir[1:])),
            "hiz_aralik": [min(olcu, default=0), max(olcu, default=0)], "satir": satir}
    (hedef / "rapor.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=1), encoding="utf-8")
    return ozet


def gecti(o: dict) -> list[str]:
    if o.get("cokme"):
        return [f"çöktü: {o['cokme']}"]
    return [f"{a} {o[a]} > {e}" for a, e in ESIK.items() if o[a] > e]


def sayfa(kok: Path, raporlar: list[dict]) -> None:
    bolum = []
    for o in raporlar:
        if o.get("cokme"):
            bolum.append(f"<h2>{html.escape(o['belge'])}</h2><p><b>çöktü:</b> {html.escape(o['cokme'])}</p>")
            continue
        sat = "".join(
            f"<tr{' class=k' if r['kalan'] or r['ctc'] else ''}><td><audio controls preload=none "
            f"src='{o['belge']}/{i:03d}.wav'></audio></td><td>{r['ton']}</td><td>{r['hiz_orani'] or '—'}</td>"
            f"<td>{html.escape(r['cumle'])}<br><small>{html.escape(r['okunus'])}"
            + (f"<br>kalan: {html.escape(' '.join(r['kalan']))}" if r["kalan"] else "")
            + (f"<br>CTC duydu: {html.escape(r['duyulan'])}" if r["ctc"] else "") + "</small></td></tr>"
            for i, r in enumerate(o["satir"]))
        bolum.append(
            f"<h2>{html.escape(o['belge'])}</h2><p><audio controls preload=none src='{o['belge']}.wav'></audio> "
            f"{o['cumle']} cümle, {o['sure_sn']} sn · kalan işaret {o['kalan']} · CTC takılan %{o['ctc'] * 100:.0f} · "
            f"hız dışı %{o['hiz_disi'] * 100:.0f} (oran {o['hiz_aralik'][0]}–{o['hiz_aralik'][1]}) · ton değişimi "
            f"{o['ton_degisimi']} · {'GEÇTİ' if not gecti(o) else 'GEÇMEDİ: ' + '; '.join(gecti(o))}</p>"
            f"<details><summary>cümleler</summary><table><tr><th></th><th>ton</th><th>hız oranı</th><th>cümle / "
            f"okunuş</th></tr>{sat}</table></details>")
    (kok / "dinle.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Makale sınavı</title><style>body{font:14px sans-serif;margin:16px;"
        "max-width:1000px}td{padding:3px 6px;border-bottom:1px solid #ddd;vertical-align:top}tr.k{background:#fde8e8}"
        "audio{height:28px;width:200px}</style><h1>Makale sınavı</h1><p>seslen-tts'in kendi yoluyla. kalan: okunuşta "
        "okunmadan kalan işaret; hız oranı: sesin doğal hızına göre (1 = doğal).</p>" + "".join(bolum), encoding="utf-8")


def main(argv: list[str]) -> int:
    from seslen_tts import Seslen
    ses = argv[argv.index("--ses") + 1] if "--ses" in argv else "zeynep"
    kok = Path(argv[argv.index("--cikti") + 1]) if "--cikti" in argv else KOK / "cikti" / "makale"
    atla = {argv[argv.index(b) + 1] for b in ("--ses", "--cikti") if b in argv}
    belgeler = [Path(a) for a in argv if not a.startswith("--") and a not in atla]
    s = Seslen(klasor=os.environ.get("SESLEN_TTS_MODEL") or KOK / "cikti" / "acik" / "seslen-tts")
    girdi = [(b.stem, b.read_text(encoding="utf-8")) for b in belgeler] or [("ornek", ORNEK)]
    raporlar = []
    for ad, metin in girdi:
        o = sinav(s, ad, metin, ses, kok / ad)
        raporlar.append(o)
        print(json.dumps({k: v for k, v in o.items() if k != "satir"}, ensure_ascii=False), "→",
              "GEÇTİ" if not gecti(o) else "GEÇMEDİ: " + "; ".join(gecti(o)), flush=True)
    sayfa(kok, raporlar)
    print(kok / "dinle.html")
    return 1 if "--esik" in argv and any(gecti(o) for o in raporlar) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
