# -*- coding: utf-8 -*-
"""
ÖĞRENCİ DİNLETİSİ — eğitilen modeli işlemcide konuşturur: dinleti/ogrenci.html

1) Stiller: her ses × duygu için eğitim kayıtlarından ADET tanesinin stil
   vektörü ortalanır → cikti/ogrenci/<önek>stiller.npz (çalışma anı + stüdyo bunu kullanır).
2) Örnekler: her ses 6 duyguda, eğitimde OLMAYAN cümlelerle; ayrıca hız (RTF) ölçülür.

    python ogrenci/dinle.py cikti/ogrenci/epoch_2nd_00000.pth
"""
from __future__ import annotations

import html
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "ogrenci"))
sys.path.insert(0, str(KOK / "ogretmen"))
from model import ORAN, Ogrenci  # noqa: E402
from yukle import sureci_hazirla  # noqa: E402

ADET = 30
DUYGU = {"notr": "nötr", "neseli": "neşeli", "heyecan": "heyecan", "uyari": "uyarı", "merak": "merak", "ozur": "özür",
         "uzgun": "üzgün", "kizgin": "kızgın", "fisilti": "fısıltı"}     # son üçü 03.10.2026'da eklendi
CUMLE = {   # eğitimde yok (derlemde / duygu havuzlarında değil)
    "notr": "Kuzey mahallesindeki yeni su hattı ölçümleri rapora eklendi.",
    "neseli": "Günaydın! Bugün hava çok güzel, haritada birlikte gezelim mi?",
    "heyecan": "İnanılmaz, Serdivan'daki arıza sadece yedi dakikada giderildi!",
    "uyari": "Dikkat, bu katmanı silersen bütün çizimlerin kaybolur.",
    "merak": "Acaba bu parselin sınırı neden bu kadar eğri çizilmiş?",
    "ozur": "Üzgünüm, aradığın adresi şu an haritada bulamadım.",
    "uzgun": "Çocukluğumun geçtiği o eski sokak artık hiç tanınmıyor.",
    "kizgin": "Bu kaçıncı gecikme, bir açıklama bekliyorum!",
    "fisilti": "Gel, sana kimsenin bilmediği bir yol göstereceğim.",
}


def stiller(o: Ogrenci) -> dict[str, np.ndarray]:
    ozet = json.loads((KOK / "veri" / "egitim" / "ozet.json").read_text(encoding="utf-8"))["duygu"]
    grup: dict[str, list[str]] = {}
    for anahtar, d in ozet.items():
        ses, kimlik = anahtar.split("/")
        if kimlik[:2] in ("ks", "ms"):        # kısa (sessizlikle 1 sn'ye tamamlanmış) ve masal kayıtları stile girmez
            continue
        grup.setdefault(f"{ses}/{d}", []).append(anahtar)
    sonuc = {}
    rnd = random.Random(3)
    for k, liste in sorted(grup.items()):
        vek = []
        for a in rnd.sample(liste, min(ADET, len(liste))):
            x, _ = sf.read(KOK / "veri" / "egitim" / "wavs" / f"{a}.flac", dtype="float32")
            vek.append(o.stil(x))
        sonuc[k] = np.mean(vek, axis=0)
    np.savez(KOK / "cikti" / "ogrenci" / f"{o.onek}stiller.npz", **sonuc)      # 3. aşama: t3-stiller.npz
    return sonuc


def main() -> int:
    sureci_hazirla()
    agirlik = sys.argv[1]
    o = Ogrenci(agirlik)
    t0 = time.time()
    st = stiller(o)
    print(f"{len(st)} stil ({time.time() - t0:.0f} sn)", flush=True)
    sesler = sorted({k.split("/")[0] for k in st}, key=lambda s: list(json.loads(
        (KOK / "veri" / "egitim" / "konusmacilar.json").read_text(encoding="utf-8"))).index(s))
    hedef = KOK / "dinleti" / "ses"
    hedef.mkdir(parents=True, exist_ok=True)
    etiket = Path(agirlik).stem
    satir, top_ses, top_is = [], 0.0, 0.0
    for ses in sesler:
        hucre = []
        for d in DUYGU:
            if f"{ses}/{d}" not in st:
                hucre.append("<td>—</td>")
                continue
            t = time.perf_counter()
            w = o.konus(CUMLE[d], st[f"{ses}/{d}"])
            top_is += time.perf_counter() - t
            top_ses += len(w) / ORAN
            ad = f"ogr-{etiket}-{ses}-{d}.wav"
            sf.write(hedef / ad, w / max(1e-6, np.abs(w).max()) * 0.9, ORAN)
            hucre.append(f"<td><audio controls preload=none src='ses/{ad}'></audio></td>")
        satir.append(f"<tr><th>{ses.capitalize()}</th>{''.join(hucre)}</tr>")
        print(f"{ses} tamam", flush=True)
    rtf = top_is / max(top_ses, 1e-6)
    e = html.escape
    cumleler = "".join(f"<li><b>{DUYGU[d]}</b>: {e(c)}</li>" for d, c in CUMLE.items())
    (KOK / "dinleti" / "ogrenci.html").write_text(f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Öğrenci dinletisi</title><style>
:root {{ --zemin:#f4f5f8; --yuzey:#fff; --metin:#1b2230; --soluk:#5f6b7c; --cizgi:#dde2ea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --zemin:#0f1216; --yuzey:#171c23; --metin:#e8edf2; --soluk:#8b98a8; --cizgi:#2a323d; }} }}
body {{ margin:0; background:var(--zemin); color:var(--metin); font:14px/1.45 system-ui, sans-serif; }}
main {{ max-width:1500px; margin:0 auto; padding:24px 16px 64px; }} p, li {{ color:var(--soluk); }}
.kaydir {{ overflow-x:auto; }} table {{ border-collapse:collapse; background:var(--yuzey); border-radius:12px; }}
th, td {{ border-bottom:1px solid var(--cizgi); padding:8px; text-align:left; }} audio {{ width:200px; height:30px; }}
</style></head><body><main><h1>Öğrenci (StyleTTS2) — {e(etiket)}</h1>
<p>Bu sesleri GPU'suz, bu bilgisayarın işlemcisi üretti (RTF {rtf:.2f}: 1 saniyelik ses {rtf:.2f} saniyede).
Cümlelerin hiçbiri eğitimde yoktu. Her hücre: o ses, o duygunun stil vektörüyle.</p><ol>{cumleler}</ol>
<div class=kaydir><table><tr><th>ses</th>{''.join(f'<th>{v}</th>' for v in DUYGU.values())}</tr>{''.join(satir)}</table></div>
</main></body></html>""", encoding="utf-8")
    print(f"dinleti/ogrenci.html yazıldı; RTF {rtf:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
