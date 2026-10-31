# Yol haritası — çok sesli, duygulu, GPU'suz öğrenci + karakter stüdyosu

> **Durum (11.10.2026).** Bu sayfa 29.09'daki plandır; aşağısı o günün tahminleriyle duruyor. Gerçekleşen
> (ayrıntı [README.md](README.md)):
> - Kadro 14 değil **16 ses** (Selin ve Ece eklendi), duygu 6 değil **9** (03.10: üzgün, kızgın, fısıltı).
> - Öğretmen verisi Kaggle (ücretsiz) ve RunPod'da (RTX 4090) üretildi. Türkçe PL-BERT ayrı projede eğitildi
>   (`Masaüstü/ses_projeleri/turkce-plbert`, 150 bin adım, tek 4090). G2P de ayrı projeye taşındı (turkce-vurgu).
> - Öğrenci başlangıç ağırlığı LibriTTS değil StyleTTS2-LJSpeech (MIT, [LISANSLAR.md](LISANSLAR.md)'de).
> - Öğrenci: 1–2. aşama (vurgusuz, İngilizce PL-BERT), 3. aşama (vurgulu, Türkçe PL-BERT, 9 ton; RTX 4090,
>   ~37 saat). Ardından kısa ince ayarlar: t4 (odak, kullanılmadı), t5 (yalnız Serdar), t6 (10–11.10: yeni Fatma,
>   kısa ve masal cümleleri; değerlendirme bekliyor).
> - Çözücü ONNX'e aktarıldı (int8 denenmedi). Seslen'e bağlama yapıldı: Seslen artık bu stüdyonun kendisi.
> - Karakter stüdyosu (karıştırma, kaydetme) bu depoda değil, `Masaüstü/cbs-karakter-studyo` projesinde.
> - Açık sürüm: `seslen-tts` (6 ses), bkz. [YAYIN.md](YAYIN.md).

Karar (29.09.2026): öğrenci **StyleTTS2** (Kokoro tipi: difüzyonsuz, iSTFTNet
çözücü), piper değil. Neden:

| | piper (VITS) | StyleTTS2 / Kokoro tipi |
|---|---|---|
| Kalite (temiz öğretmen verisiyle) | iyi telaffuz, düz tonlama, 22 kHz | ElevenLabs'e en yakın küçük model ailesi (Kokoro-82M arenada 1.) |
| Duygu | ses kimliği başına ayrı "konuşmacı"; ifade zayıflar | stil vektörü tonlamayı taşır |
| Ses çeşidi / stüdyo | yalnız eğitilmiş kimlikler, karıştırma yok | her ses 256 sayılık bir **stil vektörü**: karıştırılır, kaydırılır |
| İnce-kalın ayarı | sonradan perde kaydırma (bozar) | çözücü perde eğrisini (F0) girdi alır: ölçeklenir, temiz |
| İşlemci hızı (bu makine) | RTF ~0,05 | RTF ~0,25 (ölçüldü: ogrenci/styleHizOlc.py, git geçmişinde) → 4 sn cümle ~1 sn |

Cümle cümle akış olduğu için ilk ses ~0,5-1 sn'de başlar. int8 ONNX ile
~2× daha hızlı olabilir (denenecek).

## Kadro (ogretmen/kadro.py)

14 ses, hepsi 6 duyguyla (nötr, neşeli, heyecan, uyarı, merak, özür):

- yetişkin: Damla, Kuzey, Ayşe (olgun kadın), Serdar (olgun erkek, kalın)
- genç: Zeynep, Emre · yaşlı: Fatma nine, Hasan dede
- çocuk: Elif, Can
- çizgi film (özgün): Pıtırcık (minik tiz), Koca (iri kalın), Fırfır (kurnaz tilki), Kara (kötü büyücü)

Her sesin 3 tarif adayı → dinle, seç (bulut/kaggle/tasarim.py, dinleti/kadro.html).

## Karakter stüdyosu (hepsi işlemcide, GPU yok)

Model bir kez eğitilir; stüdyo yalnız stil vektörleriyle oynar:
- **ses seç** (14 hazır ses) + **duygu seç** (6)
- **karıştır**: "Ayşe %70 + Fatma %30" gibi iki sesin arası → yeni karakter
- **ince ↔ kalın**: perde eğrisi ölçeği · **hız** · **duygu şiddeti**
  (duygu yönü = duygulu stil − nötr stil; her sese, karışıma da eklenir)
- **kaydet**: yeni karakter birkaç KB'lık bir dosya (vektör + ayarlar)
Sınır: eğitilen seslerin "arasını" üretir. Bambaşka bir ses türü (ör. robot)
için kadroya yeni ses eklenir: veri ~1 $ + ek eğitim ~10-20 $.

## Masraf ve süre

Kiralık GPU: RunPod (güvenilir) ya da Vast.ai (ucuz). Fiyatlar 09.2026:
RTX 4090 ~0,35-0,70 $/sa, A100 80 GB ~0,8-1,4 $/sa.

| Adım | Nerede | GPU saati | Masraf |
|---|---|---|---|
| Aday sesler (36 aday) | Kaggle | ~1 | 0 |
| Öğretmen verisi: 12 yeni ses × ~2000 cümle (~21 sa ses, 2 aday) | 4090 (VoxCPM2 RTF ~0,3 bf16) | ~15-20 | ~10 $ |
| Türkçe PL-BERT (tonlama için metin modeli; Vikipedi + bizim G2P) | 4090 | ~15-25 | ~10-20 $ |
| Pilot eğitim (Damla, Kuzey + 2 ses; kalite kapısı) | A100 | ~30 | ~30-40 $ |
| Tam eğitim (14 ses, ~30 sa veri) | A100 | ~150-250 | ~150-300 $ |
| **Toplam** | | | **~200-370 $** (beklenen ~250 $) |

Tahmin tabanı: Kokoro v0.19 <100 sa veri, <20 tur, ~500 A100 saati (~400 $);
StyleTTS2 ince ayarında ~0,25-0,3 A100 saati / (ses saati × tur).
Pilot kötü çıkarsa büyük harcama yapılmaz.

Takvim (~5-6 hafta):
1. hafta: aday sesler → seçim; G2P (bitti: ogrenci/g2p.py); PL-BERT hazırlığı
2. hafta: öğretmen verisi (kiralık 4090, 1-2 gün) + PL-BERT eğitimi
3. hafta: pilot eğitim → dinleme → karar
4-5. hafta: tam eğitim, ONNX, işlemci hızı, Seslen'e bağlama (ses + duygu alanı)
5-6. hafta: stüdyo (karıştırma / perde / hız / kaydet) API + sayfa

Lisans: StyleTTS2 kodu MIT, VoxCPM2 Apache-2.0, derlem CC0, G2P bizim.
Önceden eğitilmiş StyleTTS2 LibriTTS ağırlıkları başlangıç noktası olabilir;
şartı: dinleyene sesin yapay olduğu söylenecek (asistan için zaten öyle) —
kullanmadan önce LISANSLAR.md'ye işlenecek.
