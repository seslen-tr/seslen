# -*- coding: utf-8 -*-
"""
SESLEN SUNUCUSU — stüdyo arayüzü ve API: http://127.0.0.1:8780

Yalnız bu bilgisayarda çalışır (127.0.0.1), internete çıkmaz, GPU gerekmez. Metin cümle cümle seslendirilir;
her cümle hazır olur olmaz tarayıcıya gider (NDJSON akışı), arayüz sırayla çalar.

  Arayüzün uçları:
    POST /api/seslendir   {metin, ayar}  → satır satır: {"tur":"parca", cumle, ton, yogunluk, sessizlik_ms,
                                          bilgi, wav (base64, 16 bit)} … {"tur":"son"}
    POST /api/okunus      {metin} → cümle cümle okunuş (normalleştirilmiş sözcükler, vurgulu hece)
    GET/POST/DELETE /api/telaffuz   telaffuz sözlüğü (kullanici/telaffuz.json)
    POST /api/donustur?bicim=mp3    WAV gövdesi → MP3
    GET  /api/bilgi       sesler, tonlar, üsluplar, işaretler, kalite ölçüleri, model
    GET  /api/ornek/{ses} sesin kısa tanıtım kaydı (ilk istekte üretilir, sonra diskten)

  Uygulamalar için (OpenAI'nin ses ucuyla aynı biçim, tek seferde bütün ses):
    POST /v1/audio/speech {input, voice, ton?, uslup?, hiz?, perde?, yogunluk?, tonlama?, duraklama_ms?,
                           virgul_ms?, response_format: wav | mp3} → ses dosyası
    GET  /v1/voices       sesler ve tonları

    .venv/Scripts/python.exe ogrenci/studyoSunucu.py
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

KOK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "ogretmen")]
from studyo import Ayar, Studyo, etiketsiz, okunus  # noqa: E402

GORUNEN = {"damla": "Damla", "kuzey": "Kuzey", "ayse": "Ayşe", "serdar": "Serdar", "zeynep": "Zeynep",
           "emre": "Emre", "selin": "Selin", "fatma": "Fatma nine", "hasan": "Hasan dede", "elif": "Elif",
           "can": "Can", "ece": "Ece", "pitircik": "Pıtırcık", "koca": "Koca", "firfir": "Fırfır", "kara": "Kara"}
TONLAR = {"notr": "nötr", "neseli": "neşeli", "heyecan": "heyecan", "uyari": "uyarı", "merak": "merak",
          "ozur": "özür", "uzgun": "üzgün", "kizgin": "kızgın", "fisilti": "fısıltı"}
USLUP_AD = {"haber": "haber", "bilgi": "bilgi", "talimat": "talimat", "deneme": "deneme", "anlati": "anlatı",
            "masal": "masal", "sohbet": "sohbet"}
EN_COK = 50_000           # karakter; uzun metin paragraf paragraf işlenir
# Ana sayfadaki "bileşenler ve kalite": her biri kendi projesinin ölçümünden (README'lerde ayrıntı)
BILESENLER = [
    {"ad": "turkce-normalize", "is": "Sayı, tarih, saat, para, kısaltma → okunur yazı",
     "olcu": "128/128 örnek (turkce-tts-sinav)"},
    {"ad": "turkce-uslup", "is": "Metnin türü: haber, masal, talimat, sohbet… → okuma hızı ve tonlama",
     "olcu": "5 cümlelik metinde %96,1 (BERTurk)"},
    {"ad": "turkce-duygu", "is": "Cümle başına duygu: soru → merak, uyarı → uyarı, sevinç → neşeli",
     "olcu": "Ton seçimi %85,6"},
    {"ad": "turkce-vurgu", "is": "Vurgulu okunuş: ANkara, GELmedi, İZmir",
     "olcu": "Literatür örneklerinde %97,6 (turkce-tts-sinav)"},
    {"ad": "turkce-plbert", "is": "Türkçe cümle anlama: tonlamayı bağlama göre ayarlar",
     "olcu": "Türkçe Vikipedi, 150 bin adım"},
    {"ad": "StyleTTS2 öğrencisi", "is": "16 ses × 9 duygu, işlemcide, internetsiz",
     "olcu": "Anlaşılırlık %100 · doğallık (UTMOS) 3,60"},
]
ISARETLER = [
    {"ornek": "[üzgün]", "ne": "Bundan sonraki cümleler üzgün okunur. Öteki duygular: [nötr] [neşeli] [heyecan] "
                                "[uyarı] [merak] [özür] [kızgın] [fısıltı]; geri dönmek için [otomatik]."},
    {"ornek": "[duraklama 1.5]", "ne": "1,5 saniye durur. [duraklama 800 ms] de olur; sayı yazılmazsa 0,8 sn."},
    {"ornek": "[yavaş]", "ne": "Bundan sonrası yavaş. [hızlı], [hız 1.2] ve geri dönmek için [normal]."},
    {"ornek": "boş satır", "ne": "Yeni paragraf: arada daha uzun durulur."},
]


class AyarGirdi(BaseModel):
    ses: str = "zeynep"
    ton: str = "otomatik"
    yogunluk: float | None = Field(None, ge=0, le=1.5)
    perde: float = Field(1.0, ge=0.7, le=1.4)
    uslup: str = "otomatik"
    hiz: float | None = Field(None, ge=0.6, le=1.6)
    tonlama: float | None = Field(None, ge=0, le=2)
    duraklama_ms: int | None = Field(None, ge=0, le=2000)
    virgul_ms: int | None = Field(None, ge=0, le=800)


class Istek(BaseModel):
    metin: str = Field(..., min_length=1, max_length=EN_COK)
    ayar: AyarGirdi = AyarGirdi()


class Konusma(AyarGirdi):
    input: str = Field(..., min_length=1, max_length=EN_COK)
    voice: str | None = None            # "ses" ile aynı; OpenAI istemcileri bu adı gönderir
    model: str | None = None            # OpenAI uyumluluğu için kabul edilir, kullanılmaz
    response_format: str = "wav"


class OkunusIstek(BaseModel):
    metin: str = Field(..., min_length=1, max_length=EN_COK)


class TelaffuzIstek(BaseModel):
    yazilis: str = Field(..., min_length=1, max_length=60)
    okunus: str = Field(..., min_length=1, max_length=120)


app = FastAPI(title="Seslen")
_studyo: Studyo | None = None
_kilit = threading.Lock()          # model tek; seslendirmeler sırayla
_yukle = threading.Lock()
# Seslendirme hep AYNI iş parçacığında: sunucu her cümleyi başka bir iş parçacığında çalıştırınca PyTorch her
# birinde kendini yeniden ısıtıyordu (ilk cümle 0,9 sn yerine 2,6 sn, 06.10)
_isci = ThreadPoolExecutor(max_workers=1, thread_name_prefix="seslendirme")
_BITTI = object()


def studyo() -> Studyo:
    global _studyo
    with _yukle:
        if _studyo is None:
            _studyo = Studyo()
    return _studyo


def _ayar(a: AyarGirdi) -> Ayar:
    st = studyo()
    if a.ses not in {k.split("/")[0] for k in st.stiller}:
        raise HTTPException(400, "bilinmeyen ses")
    if a.ton != "otomatik" and a.ton not in TONLAR:
        raise HTTPException(400, "bilinmeyen ton")
    if a.uslup != "otomatik" and a.uslup not in USLUP_AD:
        raise HTTPException(400, "bilinmeyen üslup")
    return Ayar(**a.model_dump())


def _bayt(x: np.ndarray, oran: int, bicim: str = "wav") -> bytes:
    b = io.BytesIO()
    if bicim == "mp3":
        sf.write(b, x, oran, format="MP3", subtype="MPEG_LAYER_III")
    else:
        sf.write(b, x, oran, format="WAV", subtype="PCM_16")
    return b.getvalue()


def _aciklama(a: str) -> str:
    """Kadro notundan kullanıcıya gösterilecek açıklama: iç notlar ("fatma-1 tasarımı") atılır, ilk harf büyük."""
    a = re.sub(r"\s*\([^)]*tasarım[^)]*\)", "", a).strip()
    return ("İ" if a[:1] == "i" else a[:1].upper()) + a[1:] if a else a


SITE = KOK / "ogrenci" / "site"
SITE_DOSYA = {"stil.css": "text/css; charset=utf-8", "uygulama.js": "text/javascript; charset=utf-8"}
# no-cache: tarayıcı her açılışta sorar (değişmediyse 304); güncellemeden sonra eski arayüz kalmasın
TAZE = {"Cache-Control": "no-cache"}


@app.get("/")
def sayfa():
    return FileResponse(SITE / "index.html", media_type="text/html; charset=utf-8", headers=TAZE)


@app.get("/site/{ad}")
def site_dosyasi(ad: str):
    if ad not in SITE_DOSYA:
        raise HTTPException(404, "yok")
    return FileResponse(SITE / ad, media_type=SITE_DOSYA[ad], headers=TAZE)


@app.get("/api/bilgi")
def bilgi():
    from kadro import KADRO
    from turkce_uslup import OKUMA
    st = studyo()
    sira = list(GORUNEN)
    sesler = sorted({k.split("/")[0] for k in st.stiller}, key=lambda s: sira.index(s) if s in sira else 99)
    return {
        "model": st.agirlik.name,
        "vurgu": st.model.vurgu,
        "en_cok": EN_COK,
        "sesler": [{"ad": s, "gorunen": GORUNEN.get(s, s.capitalize()), "tur": KADRO.get(s, {}).get("tur", ""),
                    "aciklama": _aciklama(KADRO.get(s, {}).get("aciklama", "")),
                    "tonlar": sorted(st.tonlar(s), key=lambda t: list(TONLAR).index(t) if t in TONLAR else 99)}
                   for s in sesler],
        "tonlar": TONLAR,
        "usluplar": {k: {"ad": v, **asdict(OKUMA[k])} for k, v in USLUP_AD.items()},
        "bilesenler": BILESENLER,
        "isaretler": ISARETLER,
        "telaffuz_sayisi": len(st.telaffuz.sozluk),
    }


@app.post("/api/seslendir")
def seslendir(istek: Istek):
    ayar = _ayar(istek.ayar)
    st = studyo()

    def akis():
        from model import ORAN
        with _kilit:
            try:
                uretec = st.seslendir(istek.metin, ayar)
                while (p := _isci.submit(next, uretec, _BITTI).result()) is not _BITTI:
                    x = p.ses / max(1e-6, float(np.abs(p.ses).max())) * 0.9
                    yield json.dumps({"tur": "parca", "cumle": p.cumle, "ton": p.ton, "yogunluk": p.yogunluk,
                                      "sessizlik_ms": p.sessizlik_ms, "bilgi": p.bilgi,
                                      "wav": base64.b64encode(_bayt(x, ORAN)).decode()}, ensure_ascii=False) + "\n"
            except Exception as h:      # akış başladıktan sonra HTTP hatası verilemez: satırla bildir
                yield json.dumps({"tur": "hata", "ileti": f"{type(h).__name__}: {h}"}, ensure_ascii=False) + "\n"
                return
        yield json.dumps({"tur": "son"}) + "\n"

    return StreamingResponse(akis(), media_type="application/x-ndjson")


@app.post("/api/okunus")
def okunus_ucu(istek: OkunusIstek):
    return {"cumleler": okunus(istek.metin, studyo().telaffuz)}


@app.post("/api/uslup")
def uslup_ucu(istek: OkunusIstek):
    """Otomatik üslup önizlemesi (yazarken): seslendirmede de aynı karar verilir (Studyo.okuma, ilk ~4000 karakter)."""
    from turkce_uslup import uslup
    r = uslup(etiketsiz(istek.metin)[:4000])
    return {"uslup": r.uslup, "ad": USLUP_AD.get(r.uslup, r.uslup), "gerekce": r.gerekce}


@app.get("/api/telaffuz")
def telaffuz_listesi():
    return {"sozluk": studyo().telaffuz.sozluk}


@app.post("/api/telaffuz")
def telaffuz_ekle(istek: TelaffuzIstek):
    try:
        studyo().telaffuz.ekle(istek.yazilis, istek.okunus)
    except ValueError as h:
        raise HTTPException(400, str(h))
    return {"sozluk": studyo().telaffuz.sozluk}


@app.delete("/api/telaffuz/{yazilis}")
def telaffuz_sil(yazilis: str):
    studyo().telaffuz.sil(yazilis)
    return {"sozluk": studyo().telaffuz.sozluk}


@app.post("/api/donustur")
async def donustur(istek: Request, bicim: str = "mp3"):
    if bicim != "mp3":
        raise HTTPException(400, "yalnız bicim=mp3")
    govde = await istek.body()
    if len(govde) > 400 * 1024 * 1024:
        raise HTTPException(413, "dosya çok büyük")
    try:
        x, oran = sf.read(io.BytesIO(govde), dtype="float32")
    except Exception:
        raise HTTPException(400, "gövde WAV değil")
    return Response(_bayt(x, oran, "mp3"), media_type="audio/mpeg")


def _tum(metin: str, ayar: Ayar):
    """Bütün metin tek dalga (cümleler + aralarındaki sessizlik), seslendirme iş parçacığında."""
    st = studyo()
    with _kilit:
        return _isci.submit(st.tum, metin, ayar).result()


@app.post("/v1/audio/speech")
def konusma(istek: Konusma):
    from model import ORAN
    if istek.response_format not in ("wav", "mp3"):
        raise HTTPException(400, "response_format: wav ya da mp3")
    veri = istek.model_dump(exclude={"input", "voice", "model", "response_format"})
    if istek.voice:
        veri["ses"] = istek.voice
    ayar = _ayar(AyarGirdi(**veri))
    dalga, parcalar = _tum(istek.input, ayar)
    if not len(dalga):
        raise HTTPException(400, "seslendirilecek cümle yok")
    return Response(_bayt(dalga, ORAN, istek.response_format),
                    media_type="audio/mpeg" if istek.response_format == "mp3" else "audio/wav",
                    headers={"X-Cumle-Sayisi": str(len(parcalar)), "X-Uslup": parcalar[0].bilgi["uslup"],
                             "X-Tonlar": ",".join(p.ton for p in parcalar)})


@app.get("/v1/voices")
def sesler_v1():
    b = bilgi()
    return {"voices": [{"voice_id": s["ad"], "name": s["gorunen"], "category": s["tur"], "description": s["aciklama"],
                        "tones": s["tonlar"]} for s in b["sesler"]], "model": b["model"]}


@app.get("/api/ornek/{ses}")
def ornek(ses: str):
    from model import ORAN
    st = studyo()
    if ses not in {k.split("/")[0] for k in st.stiller}:
        raise HTTPException(404, "bilinmeyen ses")
    yol = KOK / "cikti" / "ornekler" / f"{st.ses_modeli(ses).onek}{ses}.wav"
    if not yol.exists():
        yol.parent.mkdir(parents=True, exist_ok=True)
        ad = GORUNEN.get(ses, ses.capitalize())
        dalga, _ = _tum(f"Merhaba, ben {ad}. Bugün size ne okumamı istersiniz?", Ayar(ses=ses, ton="notr"))
        sf.write(yol, dalga, ORAN, subtype="PCM_16")
    return FileResponse(yol, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    from yukle import sureci_hazirla
    # Windows güç kısıtlaması kapalı + öncelik biraz yüksek: yoksa gizli pencerede 2,7× yavaş (06.10)
    sureci_hazirla()
    # Modeller açılışta yüklensin ve ısınsın (ilk istek 4,5 sn bekliyordu: üslup ve duygu modelleri tembel yükleniyor)
    for _ in studyo().seslendir("Merhaba. Bugün hava güzel mi?", Ayar()):
        pass
    uvicorn.run(app, host="127.0.0.1", port=8780, log_level="warning")
