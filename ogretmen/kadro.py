# -*- coding: utf-8 -*-
"""
KADRO — atölyenin bütün sesleri tek yerde.

Her ses VoxCPM2'nin "ses tasarımı" ile TARİFTEN doğar (kimseye ait değil,
referans kayıt yok). Bir sesin birkaç tarif ADAYI olur; dinlenip biri seçilir,
seçilenin ref.wav kaydı o sesin kimliği olur. Sonra bütün veri o kayıt
referans verilerek + duygu tarifiyle (ogretmen/duygu.py) üretilir.

Çizgi film sesleri özgün tasarımdır: bilinen bir karakteri / seslendirme
sanatçısını taklit eden tarif YAZILMAZ ("X gibi" yok).

secilen: dinleyip karar verilen aday (None = henüz seçilmedi).
"""
from __future__ import annotations

KADRO: dict[str, dict] = {
    # ── yetişkin ──────────────────────────────────────────────────────────
    "ayse": {"tur": "yetişkin", "aciklama": "olgun kadın (~45), orta-kalın, güven veren", "secilen": "ayse-3", "adaylar": {
        "ayse-1": "A mature woman in her mid forties, warm and rich medium-low voice, calm, caring and trustworthy",
        "ayse-2": "A middle-aged woman, smooth alto voice, articulate and professional like a news presenter",
        "ayse-3": "A friendly mature woman around fifty, soft slightly husky voice, motherly and reassuring"}},
    "serdar": {"tur": "yetişkin", "aciklama": "olgun erkek (~45), kalın, otoriter ama dost", "secilen": "serdar-4", "adaylar": {
        "serdar-1": "A mature man in his mid forties, deep bass-baritone voice, authoritative yet friendly, steady pace",
        "serdar-2": "A middle-aged man with a thick, powerful, resonant low voice, confident and commanding",
        "serdar-3": "A calm man around fifty, warm deep voice with slight gravel, wise and composed",
        # 2. tur (29.09): serdar-1 beğenildi ama "biraz kısık" (netlik 18,0) → aynı tarif, temiz ses vurgusu
        "serdar-4": "A mature man in his mid forties, deep bass-baritone voice, clear and smooth, not hoarse, "
                    "authoritative yet friendly, steady pace",
        "serdar-5": "A mature man in his mid forties, native Turkish speaker with natural Turkish intonation, "
                    "deep clear bass-baritone voice, clean studio recording, authoritative yet friendly",
        "serdar-6": "A calm middle-aged man, deep warm resonant voice, clean and clear tone, confident news anchor"}},
    "zeynep": {"tur": "genç", "aciklama": "genç kız (~17), ince, canlı", "secilen": "emre-3", "adaylar": {
        "zeynep-1": "A teenage girl around seventeen, light high-pitched voice, lively, cheerful and quick",
        "zeynep-2": "A young girl in her late teens, thin bright voice, curious and energetic",
        "zeynep-3": "A sweet teenage girl, soft airy high voice, friendly and a bit shy"}},
    "emre": {"tur": "genç", "aciklama": "genç erkek (~18), orta, enerjik", "secilen": "emre-1", "adaylar": {
        "emre-1": "A teenage boy around eighteen, medium bright voice, energetic, casual and friendly",
        "emre-2": "A young man of eighteen, clear light tenor voice, enthusiastic and fast-paced",
        "emre-3": "A cheerful teenage boy, slightly breaking youthful voice, playful and confident"}},
    # Kullanıcı kararı (29.09): emre-3 genç kız gibi duyuldu → Zeynep; fatma-1 genç
    # kız gibi → Selin; fatma-3 çocuk gibi → Ece. Adaylar kendi klasörlerinde kalır.
    "selin": {"tur": "genç", "aciklama": "genç kadın, yumuşak (fatma-1 tasarımı)", "secilen": "fatma-1", "adaylar": {}},
    "fatma": {"tur": "yaşlı", "aciklama": "yaşlı köylü nine (~78), alçak, hırıltılı, şefkatli", "secilen": "fatma-5", "adaylar": {
        "fatma-1": "An elderly woman around seventy, thin gentle voice with a slight tremble, kind grandmother, slow pace",
        "fatma-2": "An old woman, soft warm aged voice, affectionate and storytelling, relaxed tempo",
        # 2. tur (10.10): fatma-2 "nine gibi değil" (perde 323 Hz) → yaşlılık ipuçları açıkça; kullanıcı fatma-5'i seçti
        "fatma-4": "A very old Turkish grandmother, about eighty years old, frail aged voice with a clear "
                   "age tremble, slightly breathy and creaky, slow tender pace, telling a bedtime story",
        "fatma-5": "An elderly village grandmother in her late seventies, low-pitched raspy old woman's voice, "
                   "warm and loving, speaks slowly with soft pauses",
        "fatma-6": "An old woman of eighty with a weathered, shaky, slightly hoarse voice, native Turkish "
                   "speaker, gentle and affectionate, unhurried storytelling",
        "fatma-7": "A grandmother aged seventy-five, mature deep aged female voice with gentle crackle, "
                   "kind and wise, calm slow narration, clean recording",
        "fatma-3": "A lively elderly lady in her seventies, high thin voice, cheerful and chatty"}},
    "hasan": {"tur": "yaşlı", "aciklama": "yaşlı erkek (~70), bariton, bilge dede", "secilen": "hasan-6", "adaylar": {
        "hasan-1": "An elderly man in his seventies, deep aged voice with gentle rasp, wise grandfather, slow calm pace",
        "hasan-2": "An old man, warm low voice, slightly hoarse, kind and humorous storyteller",
        "hasan-3": "A dignified elderly man around seventy five, thick resonant voice, slow and deliberate",
        # 2. tur (11.10): hasan-3 "telaffuz sorunlu, İngilizceye kayıyor" → anadili Türkçe ve net söyleyiş açıkça
        "hasan-4": "A native Turkish elderly man around seventy five from Anatolia, warm deep aged voice, clear "
                   "Turkish articulation and natural Turkish intonation, wise grandfather, calm pace",
        "hasan-5": "An old Turkish grandfather in his seventies, slightly hoarse low voice, speaks Turkish natively "
                   "with crisp consonants and full vowels, kind and slow storytelling",
        "hasan-6": "A seventy year old Turkish man, mature baritone with a gentle age rasp, fluent native Turkish "
                   "speaker, clean studio recording, calm and wise",
        "hasan-7": "An elderly Istanbul gentleman of about seventy eight, soft deep voice, precise Turkish "
                   "pronunciation, warm and dignified, unhurried"}},
    # ── çocuk ─────────────────────────────────────────────────────────────
    "elif": {"tur": "çocuk", "aciklama": "kız çocuk (~8)", "secilen": "elif-1", "adaylar": {
        "elif-1": "A little girl about eight years old, high sweet child voice, curious and cheerful",
        "elif-2": "A young girl child, bright high-pitched voice, giggly, playful and energetic",
        "elif-3": "A small girl around seven, soft childlike voice, gentle and innocent"}},
    "can": {"tur": "çocuk", "aciklama": "erkek çocuk (~9)", "secilen": "can-2", "adaylar": {
        "can-1": "A little boy about nine years old, clear high child voice, energetic and mischievous",
        "can-2": "A young boy child, bright voice, excited and talkative, fast pace",
        "can-3": "A small boy around eight, soft child voice, curious and a bit shy"}},
    "ece": {"tur": "çocuk", "aciklama": "küçük kız, çok ince (fatma-3 tasarımı)", "secilen": "fatma-3", "adaylar": {}},
    # ── çizgi film (özgün tasarım) ────────────────────────────────────────
    "pitircik": {"tur": "çizgi", "aciklama": "minik sevimli yaratık, çok ince tiz, cıvıl cıvıl", "secilen": "pitircik-3", "adaylar": {
        "pitircik-1": "A tiny cute cartoon creature, very high squeaky voice, super fast and bubbly, animated",
        "pitircik-2": "A small magical cartoon sprite, tiny high chirpy voice, giggly and hyperactive",
        "pitircik-3": "A cute little cartoon mouse character, thin squeaky voice, excited and adorable"}},
    "koca": {"tur": "çizgi", "aciklama": "iri kalın dev, gürleyen ama sevimli", "secilen": "koca-2", "adaylar": {
        "koca-1": "A big friendly cartoon giant, very deep booming voice, slow and goofy, lovable",
        "koca-2": "A large clumsy cartoon bear character, thick rumbling low voice, warm and silly",
        "koca-3": "A gentle cartoon ogre, huge deep gravelly voice, slow, kind and funny"}},
    "firfir": {"tur": "çizgi", "aciklama": "kurnaz tilki, burnundan, cilveli", "secilen": "firfir-2", "adaylar": {
        "firfir-1": "A sly cartoon fox character, nasal medium voice, sneaky, playful and theatrical",
        "firfir-2": "A cunning cartoon trickster, smooth sly voice with a nasal twist, witty and mischievous",
        "firfir-3": "A dramatic cartoon cat character, silky high nasal voice, sassy and expressive"}},
    "kara": {"tur": "çizgi", "aciklama": "kötü büyücü, kısık hırıltılı, dramatik", "secilen": "kara-1", "adaylar": {
        "kara-1": "A cartoon villain wizard, low raspy sinister voice, dramatic and theatrical, slow menacing pace",
        "kara-2": "An evil cartoon sorcerer, crackly old whispery voice, cunning and eerie",
        "kara-3": "A grumpy cartoon villain, deep growly voice, over-the-top dramatic and funny"}},
}


def _ozel() -> dict:
    """Ürün sesleri (Damla, Kuzey) açık depoda yok: tarifleri kullanici/ozel_sesler.json'da (git dışında). Dosya varsa
    kadronun başına eklenir; yoksa atölye bu iki ses olmadan çalışır."""
    import json
    from pathlib import Path
    yol = Path(__file__).resolve().parents[1] / "kullanici" / "ozel_sesler.json"
    return json.loads(yol.read_text(encoding="utf-8")) if yol.exists() else {}


OZEL = _ozel()
KADRO = {**OZEL.get("kadro", {}), **KADRO}

# Aday dinletisinde her adayın okuduğu cümleler: 0 = tarifle tasarım (ref.wav),
# kalanlar ref + duygu tarifiyle — sesin duygu aralığı da aynı anda duyulsun.
DENEME = [
    ("notr", "Merhaba! Ben buradayım, sana haritada yardımcı olabilirim."),
    ("neseli", "Harika, aradığın parseli buldum, hemen gösteriyorum!"),
    ("uyari", "Dikkat! Bu işlem geri alınamaz, emin misin?"),
    ("merak", "Peki hangi mahalleye bakmak istersin?"),
    ("notr", "Bir varmış, bir yokmuş. Evvel zaman içinde, küçük bir köyde meraklı bir kız yaşarmış."),  # 10.10: nine/masal
    ("notr", "Evladım, acele etme. Her işin bir vakti vardır; sabreden derviş muradına ermiş."),       # 11.10: dede
]


def secilenler() -> dict[str, str]:
    """{ses: seçilen aday} — yalnız karar verilmiş sesler."""
    return {ad: s["secilen"] for ad, s in KADRO.items() if s["secilen"]}


def tarif(aday: str) -> str:
    """Adayın tarifi — aday başka bir sesin listesinde olabilir (emre-3 → Zeynep)."""
    return next(s["adaylar"][aday] for s in KADRO.values() if aday in s["adaylar"])


# "Kimlik tazeleme" (bulut/kaggle/tazele.py): seçilen sesin kısa ref.wav'ı ile
# bu uzun, sakin paragraf okunur → ~10 sn'lik yeni kimlik kaydı (ref2.wav).
# Uzun ve doğal Türkçe referans: tını daha kararlı, tonlama daha Türkçe.
TAZELE_METIN = ("Merhaba, ben buradayım. Bugün haritada birlikte çalışacağız; parselleri, yolları ve "
                "mahalleleri tek tek inceleyip sana en doğru bilgiyi vereceğim.")
# ÜRETİM KİPİ (tazeleme turu sonucu, 30.09.2026; dinleti/tazele.html). Kimlik
# her seste ref2.wav (cikti/tazele/<aday>/ref2.wav), adım 10.
#   duygulu cümle → B (yalnız referans + duygu tarifi). C'de tarif sesli okunuyor.
#   nötr cümle    → sese göre B ya da C (devam kipi: özel adları ve ekleri daha
#                   doğru okuyor — "Sapanca", "şemsiyeni" — ama bazen cümle başına
#                   hece sızdırıyor; kapı + 2 aday bunları eler).
# Seçim kuralı (kullanıcı: "Whisper'a takılmayanlar arasında sen seç"): nötr
# cümlelerde B ile C'den (kapıdan geçen, doğruluk, netlik) sırasıyla iyi olan.
# Damla: kullanıcı da C'yi daha iyi buldu.
KIP_NOTR = {"damla": "C", "kuzey": "C", "ayse": "C", "serdar": "B", "zeynep": "C", "emre": "B", "selin": "B",
            "fatma": "C", "hasan": "B", "elif": "B", "can": "C", "ece": "C", "pitircik": "B", "koca": "C",
            "firfir": "C", "kara": "C"}

# Karşılaştırma cümleleri: ğ, ş, ö, ü, sayı ve bir duygu.
KARSILASTIRMA = [
    ("notr", "Sapanca bölgesinde üç yüz kırk yedi abone kaydı buldum."),
    ("notr", "Öğleden sonra yağmur bekleniyor, şemsiyeni yanına almayı unutma."),
    ("neseli", "Harika haber, arıza on iki dakikada giderildi!"),
]
