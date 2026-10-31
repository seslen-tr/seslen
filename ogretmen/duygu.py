# -*- coding: utf-8 -*-
"""
DUYGU — öğretmen verisinde her cümlenin hangi duyguyla okunacağı.

VoxCPM2 "controllable cloning": referans sesin tınısı korunur, metnin başına
eklenen İngilizce tarif tonlamayı yönlendirir: "(cheerful…)Merhaba!".
Deneme (ogretmen/duyguDene.py — git geçmişinde, 29.09.2026): tarif sesli okunmuyor, perde /
hız / tonlama belirgin değişiyor; kullanıcı altısını da beğendi.

Öğrenci bu etiketle eğitilir (piper'da her karakter × duygu ayrı "konuşmacı"
kimliği olur: hız değişmez). Çalışma anında LLM cevabına etiket koyar.

Seçim deterministik (crc32): aynı cümle her makinede aynı duyguyu alır.
  soru (?)           → merak
  ünlem (!)          → neşeli / heyecan (yarı yarıya)
  anahtar sözcük     → uyarı (dikkat, uyarı, tehlike…) / özür (üzgünüm, maalesef…)
  düz cümle          → nötr, neşeli, heyecan, uyarı, özür eşit (%20'şer)
Eşit pay: her duygu "çok kaliteli" olsun diye az verili duygu kalmasın
(29.09.2026; önce nötr %50'ydi — nötr verisi zaten etiketsiz partiden bol).
Düz cümlelerde anlamla duygu her zaman uyuşmaz; öğrenci için sorun değil —
öğrendiği şey etiketin SESTEKİ karşılığı, anlamı LLM seçecek.
"""
from __future__ import annotations

import re
import zlib

TARIF = {
    "notr": "",
    "neseli": "cheerful, happy and warm tone, smiling",
    "heyecan": "excited and enthusiastic, slightly faster",
    "uyari": "serious, firm and alert tone, clear emphasis",
    "merak": "curious, gently questioning tone",
    "ozur": "apologetic, soft and sympathetic tone, slower",
}
# 03.10.2026: üzgün / kızgın / fısıltı. Aday tarifler GPU'da denendi
# (veri/plan_deneme, plan_deneme2); ölçüme göre iyisi asıl ada atandı (aşağıda).
ADAY_TARIF = {
    "uzgun_a": "sad and sorrowful tone, low energy, slower, slightly trembling voice",
    "uzgun_b": "melancholic and heartbroken, quiet and slow, sighing",
    # _a / _b perdeyi YÜKSELTTİ (+4 yarım ton, ağlamaklı); tipik üzüntüde perde düşer → iki aday daha
    "uzgun_c": "sad and tired, low pitch, quiet and slow voice",
    "uzgun_d": "deeply sad, low and soft voice, slow and subdued",
    "kizgin_a": "angry and irritated tone, sharp, loud and forceful",
    "kizgin_b": "furious and frustrated, scolding, tense voice",
    "fisilti_a": "whispering, very quiet breathy whisper",
    "fisilti_b": "soft whisper, secretive and hushed, close to the microphone",
}
TARIF |= ADAY_TARIF
# Seçim (03.10, ogretmen/yeniDuyguOlc.py — git geçmişinde, 4 ses, nötre göre): uzgun_c perde −2,3 yt, hız −1,5 hece/sn,
# enerji −1,3 dB; kizgin_a enerji +2,9 dB, perde +2,9 yt; fisilti_a enerji −5,1 dB, sesli oran −0,19.
# Hepsinde Whisper doğruluğu 1,00.
TARIF |= {"uzgun": ADAY_TARIF["uzgun_c"], "kizgin": ADAY_TARIF["kizgin_a"], "fisilti": ADAY_TARIF["fisilti_a"]}
# 08.10.2026: Serdar'da (kalın, otoriter kimlik kaydı) neşeli ve uyarı nötrden zor ayrılıyordu; güçlü tarif
# (ogretmen/serdarDene.py, dinleti/serdar.html) perde farkını ~2 katına çıkardı. Yalnız Serdar'ın yeniden üretiminde
# (o günkü serdarYeniden.py, bugün bulut/kiralik/yenile.py) kullanılır; neşeli cfg 3 ile, uyarı cfg 2 ile (cfg 3'te bağırmaya kaçtı).
TARIF |= {
    "neseli_guclu": "very happy and joyful, bright smiling voice, lively and energetic, higher pitch, expressive melody",
    "uyari_guclu": "urgent and alarmed warning, loud, tense and commanding, emphatic, raised voice",
}
_UYARI = re.compile(r"\b(dikkat|uyarı|tehlike|yasak|asla|sakın|acil)", re.I)
_OZUR = re.compile(r"\b(üzgünüm|maalesef|özür|kusura|ne yazık)", re.I)
_DUZ = (("notr", 20), ("neseli", 20), ("heyecan", 20), ("uyari", 20), ("ozur", 20))


def sec(kimlik: str, metin: str) -> str:
    m = metin.strip()
    if _OZUR.search(m):
        return "ozur"
    if _UYARI.search(m):
        return "uyari"
    z = zlib.crc32(kimlik.encode()) % 100
    if m.endswith("?"):
        return "merak"
    if m.endswith("!"):
        return "neseli" if z < 50 else "heyecan"
    for ad, pay in _DUZ:
        if z < pay:
            return ad
        z -= pay
    return "notr"


def metne_ekle(duygu: str, metin: str) -> str:
    t = TARIF[duygu]
    return f"({t}){metin}" if t else metin
