# -*- coding: utf-8 -*-
"""
AÇIK MODEL ÖLÇÜMÜ VE ÖRNEKLERİ — acikAktar.py'nin çıktısını seslen-tts paketinin KENDİ yoluyla (PyTorch'suz, ONNX)
seslendirir: her ses × 14 sınama cümlesi (sayı, saat, para, soru, ünlem, uzun cümle). Ölçü: CTC okunuş denetimi
(araclar/olcu/denetim.py, takılan cümle), UTMOS, gerçek zaman oranı. Her sesten bir tanıtım örneği yazar
(model klasörü/ornekler/<ses>.wav; Hugging Face kartı bunları çalar) ve dinleme sayfası kurar.

    python ogrenci/aktar/acikOlc.py [ses …]     → cikti/acik/olcum/ (olcum.json, dinle.html), ornekler/
seslen-tts deposu: SESLEN_TTS_DEPO ortam değişkeni ya da Masaüstü/ses_projeleri/seslen-tts.
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
MODEL = KOK / "cikti" / "acik" / "seslen-tts"
CIKTI = KOK / "cikti" / "acik" / "olcum"
sys.path[:0] = [str(DEPO), str(KOK / "araclar" / "olcu")]

CUMLELER = [
    "Tamam.",
    "Bugün hava çok güzel, parka gidelim mi?",
    "Toplantı saat 09:45'te başlayacak.",
    "Siparişinizin tutarı 1.250,75 TL.",
    "Dikkat! Kapılar kapanıyor.",
    "Ankara'dan gelen ekip, yarın sabah saat dokuzda İzmir'e geçecek; rapor da o zaman tamamlanacak.",
    "Üç yüz kırk yedi abone kaydı bulundu.",
    "Lütfen 15 Mart 2026 tarihine kadar başvurunuzu tamamlayın.",
    "Neden bu kadar geç kaldın?",
    "Çok teşekkür ederim, gerçekten büyük bir iyilik yaptınız.",
    "Maalesef aradığınız numaraya şu anda ulaşılamıyor.",
    "Kırmızı başlıklı kız, ormanın derinliklerinde yürürken bir kurtla karşılaşmış.",
    "Yüzde 85 oranında başarı sağlandı.",
    "Hoş geldiniz, size nasıl yardımcı olabilirim?",
]
# Tanıtım örnekleri (ton otomatik: turkce-duygu cümleye göre seçer). İlk altısı 0.2'deki örneklerle aynı metin.
ORNEK = {
    "ayse": "Hoş geldiniz, ben Ayşe. Size nasıl yardımcı olabilirim? Randevunuz yarın saat 10:30'da.",
    "zeynep": "Harika bir haberim var! Sınavı geçtim! Akşam bize gelecek misin?",
    "emre": "Dikkat edin, yol çok kaygan! Lütfen yavaş sürün.",
    "fatma": "Bir varmış, bir yokmuş. Evvel zaman içinde, küçük bir köyde meraklı bir kız yaşarmış.",
    "hasan": "Evladım, acele etme. Her işin bir vakti vardır.",
    "koca": "Kim var orada? Benim ormanımda ne arıyorsun?",
    "nehir": "Merhaba, ben Nehir. Toplantınız saat 14:00'te başlıyor; dosyaları sizin için hazırladım.",
    "ege": "Merhaba, ben Ege. Bu akşam size sakin bir yolculuğun hikâyesini anlatacağım.",
    "serdar": "Arkadaşlar, toplantıya başlıyoruz. Lütfen telefonlarınızı sessize alın.",
    "selin": "Bugün kendine biraz zaman ayır. Derin bir nefes al ve rahatla.",
    "elif": "Anne, bak! Bahçede kocaman bir kelebek var! Yakalayabilir miyim?",
    "can": "Bugün okulda resim yaptık. Benimki en güzeliydi, öğretmenim de çok beğendi!",
    "ece": "Bu oyuncak ayı benim en iyi arkadaşım. Adı Pamuk.",
    "pitircik": "Yaşasın! Bugün ormanda kocaman bir şenlik var! Sen de gelecek misin?",
    "firfir": "Ay, ne kadar da güzel bir peynir o öyle. Bana biraz versene, dostum?",
    "kara": "Sonunda büyülü kitap benim elimde! Artık kimse beni durduramaz!",
}


def main(secili: list[str]) -> None:
    import denetim
    import soundfile as sf
    from dogallik import puan
    from seslen_tts import Seslen, wav_yaz
    from seslen_tts.model import ORAN
    from turkce_vurgu import normallestir
    s = Seslen(klasor=MODEL)
    sesler = secili or list(s.sesler)
    yol = CIKTI / "olcum.json"
    olcum = json.loads(yol.read_text(encoding="utf-8")) if yol.exists() else {}
    (MODEL / "ornekler").mkdir(exist_ok=True)
    for ses in sesler:
        (CIKTI / ses).mkdir(parents=True, exist_ok=True)
        satir, sure, uretim = [], 0.0, 0.0
        for i, c in enumerate(CUMLELER):
            t = time.perf_counter()
            w = s.tum(c, ses=ses)
            uretim += time.perf_counter() - t
            sure += len(w) / ORAN
            p = CIKTI / ses / f"{i:02d}.wav"
            sf.write(p, w, ORAN)
            ref = " ".join(re.findall(r"[a-zçğıöşüâîû]+", normallestir(c)))
            d = denetim.denetle(w, ORAN, ref)
            satir.append({"cumle": c, "takildi": bool(d["bayrak"]), "duyulan": d.get("duyulan", ""),
                          "utmos": round(puan(p), 2)})
        if ses in ORNEK:
            wav_yaz(MODEL / "ornekler" / f"{ses}.wav", s.tum(ORNEK[ses], ses=ses))
        olcum[ses] = {"gorunen": s.sesler[ses]["gorunen"], "aciklama": s.sesler[ses]["aciklama"],
                      "utmos": round(float(np.mean([r["utmos"] for r in satir])), 2),
                      "takilan": sum(r["takildi"] for r in satir), "cumle": len(satir),
                      "rtf": round(uretim / max(sure, 1e-6), 2), "satir": satir}
        print(ses, {k: v for k, v in olcum[ses].items() if k != "satir"}, flush=True)
        yol.write_text(json.dumps(olcum, ensure_ascii=False, indent=1), encoding="utf-8")
    sayfa(olcum)


def sayfa(olcum: dict) -> None:
    bolum = []
    for ses, o in olcum.items():
        cumle = "".join(f"<li><audio controls preload=none src='{ses}/{i:02d}.wav'></audio> {html.escape(r['cumle'])}"
                        + (f" <b>takıldı</b> <small>(duyulan: {html.escape(r['duyulan'])})</small>" if r["takildi"]
                           else "") + "</li>" for i, r in enumerate(o["satir"]))
        ornek = f"../seslen-tts/ornekler/{ses}.wav"
        bolum.append(f"<section><h2>{html.escape(o['gorunen'])} <small>({ses})</small></h2><p>{html.escape(o['aciklama'])}"
                     f" · UTMOS {o['utmos']} · takılan {o['takilan']}/{o['cumle']} · gerçek zaman oranı {o['rtf']}</p>"
                     f"<p>Tanıtım: <audio controls preload=none src='{ornek}'></audio> "
                     f"{html.escape(ORNEK.get(ses, ''))}</p><details><summary>14 sınama cümlesi</summary><ol>{cumle}"
                     f"</ol></details></section>")
    (CIKTI / "dinle.html").write_text(
        "<!doctype html><meta charset=utf-8><title>seslen-tts sesleri</title><style>body{font:14px sans-serif;"
        "margin:16px;max-width:900px}audio{height:30px;vertical-align:middle;width:220px}section{border-bottom:1px "
        "solid #ddd;padding:6px 0}li{margin:4px 0}</style><h1>seslen-tts: açık modelin sesleri</h1><p>Paketin kendi "
        "yoluyla (ONNX, PyTorch'suz), ton otomatik. 'takıldı': Türkçe CTC denetimi bir sözcüğü farklı duydu.</p>"
        + "".join(bolum), encoding="utf-8")
    print(CIKTI / "dinle.html")


if __name__ == "__main__":
    main(sys.argv[1:])
