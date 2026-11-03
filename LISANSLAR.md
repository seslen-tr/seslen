# Lisanslar

Satılacak bir ürüne gidecek her parça ticari kullanıma açık olmalı.

| Parça | Lisans | Ticari | Not |
|---|---|---|---|
| VoxCPM2 (openbmb) — öğretmen | Apache-2.0 | evet | Kullanım koşulu: taklit, dolandırıcılık, dezenformasyon yasak; yapay zekâ sesi olduğu belirtilmeli |
| ISSAI Turkish Speech Corpus | CC BY 4.0 (makale/README); HF etiketi MIT | evet | Kaynak gösterilmeli. 16 kHz: yalnız telaffuz/temel model için, son ses kalitesi için değil. Kayıttaki gerçek kişilerin sesi TAKLİT EDİLMEZ |
| YODAS2 Türkçe (espnet/yodas2; Creative Commons lisanslı YouTube videoları) | CC BY 3.0 | evet | t8–t10 öğrencisinde telaffuz için yardımcı veri: t8–t9 ön eğitimi ~22 sa, t10 listesinde 1.000 kesit. Seslerin hiçbiri bu konuşmacılardan değildir (16 ses öğretmenin tarif tasarımı); kayıttaki gerçek kişilerin sesi TAKLİT EDİLMEZ. Atıf: Li vd. 2023, "YODAS: YouTube-Oriented Dataset for Audio and Speech" |
| Common Voice Türkçe cümle havuzu (sentence-collector, wiki.tr) — derlem metni | CC0 | evet | Yalnız METİN; Common Voice ses kayıtları kullanılmıyor |
| faster-whisper / Whisper medium | MIT | evet | Yalnız ölçüm ve doğrulama |
| Türkçe CTC tanıyıcı (mpoyraz/wav2vec2-xls-r-300m-cv8-turkish; XLS-R 300M + Common Voice 8.0 Türkçe) | Apache-2.0 | evet | Okunuş denetimi ve özdenetim. seslen-tts'in özdenetimi için ONNX'e aktarılıp 8 bite indirilmiş hâli (`denetim.onnx`, `ogrenci/aktar/denetimAktar.py`) model deposunda isteğe bağlı dosya olarak dağıtılır; atıf `denetim.json`'da ve model kartında |
| StyleTTS2 (yl4579) — öğrenci kodu ve başlangıç ağırlığı (StyleTTS2-LJSpeech) | MIT | evet | LJSpeech ağırlığı yalnız başlangıç; ses tamamen öğretmen verisiyle yeniden öğrenildi |
| Türkçe PL-BERT (turkce-plbert) | kod MIT, ağırlıklar CC BY-SA 4.0 | evet | Türkçe Vikipedi metniyle eğitildi; başlangıç modeli MIT; ayrıntı kendi README'sinde |
| turkce-duygu modeli (BERTurk + XED) | kod MIT, ağırlıklar CC BY 4.0 | evet | Atıf: Öhman vd. 2020 (XED); BERTurk MIT; `fiiller.txt` Vikisözlük türevi (CC BY-SA 4.0) |
| turkce-uslup | kod MIT, modeller CC BY-SA 4.0 | evet | UD Turkish-BOUN ve Vikipedi türevi; BERTurk MIT |
| turkce-vurgu | kod MIT, sözlükler CC BY-SA 4.0 | evet | Sözlükler Vikisözlük türevi |
| turkce-normalize | MIT | evet | |
| Açık model seslen-tts (bu modelin 16 sesi) | kod MIT, ağırlık ve sesler CC BY-SA 4.0 | evet | Yapay ses olduğu belirtilmeli, taklit yasak (kendi README'si) |

KULLANILMAYACAKLAR (ağırlıkları ticari kullanıma kapalı): XTTS-v2, F5-TTS ve
Türkçe ince ayarları (Anka-TTS vb.), Fish Audio S1/S2, Higgs Audio v3,
Voxtral TTS. ElevenLabs / Azure / Google çıkışlarıyla eğitim yapılmaz
(kullanım şartları çıkışla model eğitmeyi kısıtlar).
