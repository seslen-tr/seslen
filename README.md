# Seslen

Türkçe, duygulu, çok sesli seslendirme. 16 ses × 9 duygu. GPU ve internet gerekmez: her şey bu bilgisayarın
işlemcisinde çalışır, metin ve ses dışarı çıkmaz.

Kardeş projeler (`Masaüstü/ses_projeleri/`) burada birleşir: metin hazırlama
([turkce-normalize](../../ses_projeleri/turkce-normalize)), vurgu ([turkce-vurgu](../../ses_projeleri/turkce-vurgu)),
Türkçe PL-BERT ([turkce-plbert](../../ses_projeleri/turkce-plbert)), duygu seçimi
([turkce-duygu](../../ses_projeleri/turkce-duygu)), üslup ([turkce-uslup](../../ses_projeleri/turkce-uslup)); ölçüm için
[turkce-tts-sinav](../../ses_projeleri/turkce-tts-sinav). Bu modelin açık, PyTorch'suz sürümü
[seslen-tts](../../ses_projeleri/seslen-tts)'tir (6 açık ses; `ogrenci/acikAktar.py` hazırlar, bkz. [YAYIN.md](YAYIN.md)).

> **07.10.2026:** Seslen artık Ses Atölyesi'nin kendisidir. Eski Seslen (Piper öğrencisi + WORLD çizgi dönüşümü,
> CBS Portal'ın çağırdığı servis, port 8770) git geçmişinde f952f66'dan önce durur; büyük dosyaları o gün
> `Masaüstü/arsiv/seslen-eski` klasörüne alınmıştı (11.10'da Masaüstü'nde bu klasör yok). CBS Portal'ın `/api/seslen`
> vekili (yalnız damla/kuzey) ve gis-server hâlâ eski servisin adresine bakar (`SESLEN_BASE_URL`, varsayılan
> `http://127.0.0.1:8770`); bu depo o servisi sunmaz. Karakter tasarımı (ses karıştırma, karakter kaydetme)
> `Masaüstü/cbs-karakter-studyo` projesine taşındı: o proje bu deponun modelini koduna dokunmadan kullanır
> (`ses/sunucu.py`, `SESLEN_KOK`, http://127.0.0.1:8790, eski Seslen çağrılarıyla uyumlu `/v1/audio/speech`).

## Kullanım

`seslen.cmd` dosyasına çift tıklayın. Tarayıcıda http://127.0.0.1:8780 açılır (yalnız bu bilgisayar).

- **Metin** yazın, **Seslendir**'e basın (Ctrl+Enter). Cümleler hazır oldukça çalınır (ilk ses ~1 sn).
  Her cümlenin duygusu ve neden seçildiği listede görünür. Uzun metin (en çok 50.000 karakter) paragraf
  paragraf işlenir; boş satır yeni paragraf, arası daha uzun durulur.
- **Ses:** 16 sesten biri ("Sesler" sayfasında dinlenir). **Duygu:** otomatik (cümleye göre) ya da bütün metin
  tek duyguda. **Üslup:** otomatik ya da haber, bilgi, talimat, deneme, anlatı, masal, sohbet.
  Hız, perde ve duygu şiddeti kaydırıcıyla; tonlama ve duraklamalar "Gelişmiş ayarlar"da.
- **Metin içi işaretler** (araç çubuğundan da eklenir):

  | İşaret | Etkisi |
  |---|---|
  | `[üzgün]`, `[neşeli]`, `[nötr]`, `[heyecan]`, `[uyarı]`, `[merak]`, `[özür]`, `[kızgın]`, `[fısıltı]` | Bundan sonraki cümleler o duyguda; `[otomatik]` geri döner |
  | `[duraklama 1.5]`, `[duraklama 800 ms]`, `[duraklama]` | O noktada durur (sayısız: 0,8 sn) |
  | `[yavaş]`, `[hızlı]`, `[hız 1.2]`, `[normal]` | Bundan sonrasının hızı |

  Tanınmayan işaret okunmaz, oynatıcıda uyarı olarak gösterilir.
- **Okunuşu göster:** metnin nasıl okunacağı: açılmış sayı ve kısaltmalar, vurgulu heceler (turkce-normalize +
  turkce-vurgu).
- **Telaffuz sözlüğü:** yanlış okunan ad ya da kısaltmayı bir kez düzeltin ("GIS" → "ci ay es"); her yerde,
  API'de de geçerli. Yazılış büyük-küçük harf ve I/ı farkı gözetmeden, bütün sözcük olarak eşlenir; kesme
  işaretinden sonraki ek korunur. Sözlük `kullanici/telaffuz.json` (git dışında).
- **Özdenetim** (`ogrenci/ozdenetim.py`): her cümle üretilince Türkçe CTC tanıyıcıyla dinlenir. Bozuk sözcük,
  sonda/başta fazladan hece ya da okunmayan sözcük varsa onarımlar denenir (sözcüğü yavaşlatma, yanlış duyulan
  ünlüyü uzatma, vurguyu kaldırma, önüne duraklama, nötr ton, cümleyi yavaşlatma) ve en az kusurlu olan seçilir.
  İşe yarayan sözcük onarımı ses için `kullanici/oto_telaffuz.json`'a öğrenilir; onarılamayanlar
  `cikti/ozdenetim/duzelmeyen.jsonl`'a yazılır (sonraki eğitimin verisi). `SESLEN_OZDENETIM=0` ile kapalı.
- **Öğretmen kaydı** (`ogrenci/kayit.py`): kendiliğinden dolan, öğretmen kalitesinde kayıt. Stüdyo bir cümleyi
  (ya da ardışık cümle grubunu) önce kayıtta arar, varsa öğretmenin (VoxCPM2, 1. yol) sesini anında çalar; yoksa
  öğrenci okur ve cümle sıraya yazılır. İşçi (`python ogrenci/kayit.py isci`) sırayı işlemcide arka planda öğretmenle
  üretir; metin okunur hâle getirilir, her aday özdenetimden geçer, geçen kayda girer. Aynı konuşmada öğretmen ↔
  öğrenci geçişi kulakla fark edilmiyor. Sıra boşken işçi önceden doldurur: Seslen'e bağlanan her proje kendi
  cümlelerini `kullanici/on_kayit/<proje>.tsv` (git dışında; satır `duygu<TAB>cümle`, `{ad}` sesin adı) olarak
  bırakır, bütün seslerle üretilir; depodaki genel listeler (kısa ünlemler, masal) da girer. Kayıt
  `SESLEN_KAYIT_GB`'a (5) varınca önceden doldurma durur. Kayıt `kullanici/ogretmen_kaydi/` (git dışında). Hız ya
  da perde elle değiştirilince kayıt kullanılmaz.
- **İndir:** WAV ya da MP3.
- **API:** `POST /v1/audio/speech` (OpenAI ses ucu biçimi, WAV/MP3), `GET /v1/voices`, `POST /api/okunus`,
  `POST /api/uslup` (yazarken otomatik üslup önizlemesi), telaffuz uçları; ayrıntı ve örnekler sitenin "API"
  sayfasında ve `ogrenci/studyoSunucu.py` başında.

Kodla:

```python
from studyo import Ayar, Studyo          # ogrenci/ yolda olmalı
st = Studyo()                            # ürün ağırlığı: cikti/ogrenci/t10-epoch_2nd_00000.pth
for p in st.seslendir("[neşeli] Günaydın! [duraklama 1] [üzgün] Ama hava yağmurlu.", Ayar(ses="zeynep")):
    oynat(p.ses)                         # 24 kHz; ardından p.sessizlik_ms bekle
```

## Bir cümlenin yolu

| Sıra | Katman | Ne yapar |
|---|---|---|
| 1 | turkce-uslup | Metnin türü → hız, tonlama, duraklama |
| 2 | turkce-duygu | Cümle başına ton ve yoğunluk (konuşma eylemi kuralları + BERTurk stil çıkışı) |
| 3 | Kullanıcı ayarları | 1 ve 2'nin üstüne yazar |
| 3b | Metin içi işaretler, telaffuz sözlüğü | Hepsinin üstüne yazar; sözlük yalnız modele giden metne uygulanır |
| 4 | turkce-normalize + turkce-vurgu | Sayı, tarih, kısaltma → yazı; vurgu işaretli okunuş (`anˈtaɫjada`) |
| 5 | StyleTTS2 metin tarafı (PyTorch) | Türkçe PL-BERT, süre ve perde tahmini (~40 ms) |
| 6 | StyleTTS2 çözücü (ONNX) | Ses dalgası; perde ve tonlama burada uygulanır |

Ton, sesin stil vektörüdür (256 sayı: 128 akustik + 128 tonlama). Her ses × ton için eğitim
kayıtlarının ortalamasından bir kez çıkarılır (`cikti/ogrenci/t10-stiller.npz`); stüdyoda cümleye özgü difüzyon stiliyle karıştırılır. Yoğunluk nötrden tona
doğru kaydırmadır.

## Model

- **Sesler:** yetişkin Damla, Kuzey, Ayşe, Serdar · genç Zeynep, Emre, Selin · yaşlı Fatma nine,
  Hasan dede · çocuk Elif, Can, Ece · çizgi film Pıtırcık, Koca, Fırfır, Kara. Hepsi tariften
  tasarlandı ([ogretmen/kadro.py](ogretmen/kadro.py)); gerçek bir kişinin sesi değildir.
- **Tonlar:** nötr, neşeli, heyecan, uyarı, merak, özür, üzgün, kızgın, fısıltı.
- **Öğretmen:** VoxCPM2 (ses tasarımı + ton tarifleri). Her kayıt Whisper ile denetlendi; yanlış
  okunanlar atıldı. **Öğrenci verisi:** 3. aşamada 41,4 saat, 40.910 eğitim + 405 doğrulama kaydı; 10.10
  yenilemesinden sonra (`veri/egitim/`) 42,7 saat, 43.062 eğitim + 426 doğrulama kaydı.
- **Öğrenci:** StyleTTS2 (iSTFTNet çözücü, çalışmada difüzyon yok), LJSpeech ağırlığından ince ayar,
  kiralık GPU'da (RunPod) iki adımda:

  | Aşama | Okunuş | PL-BERT | Ton | Ağırlık |
  |---|---|---|---|---|
  | 1–2 | vurgusuz | İngilizce (özgün) | 6 | `epoch_2nd_00000…00004.pth` |
  | 3 | vurgulu (turkce-vurgu) | Türkçe (turkce-plbert, 150 bin adım) | 9 | `t3-epoch_2nd_00000…00002.pth` |

  3. aşama: 3 tur, RTX 4090, ~37 saat (04–06.10.2026). Son tur doğrulama kaybı 0,299.
- **Çalışma dosyaları** (`cikti/ogrenci/`, depoda yok): `t10-epoch_2nd_00000.pth` (ağırlık),
  `t10-stiller.npz` (ses × ton vektörleri), `t10-ses.onnx` (çözücü), `t10-config_tr.yml`.
  `model.Ogrenci` önekten (`t10-`) vurgulu okunuşu, vektör ve ONNX dosyasını kendisi seçer. Stüdyo ürün
  ağırlığını (`studyo.URUN_AGIRLIK`) kullanır. Eski turların ağırlıkları (t3-t9) 31.10'da silindi.
- **Veri yenileme ve t6** (10–11.10.2026, [bulut/kiralik/yenile.py](bulut/kiralik/yenile.py)): Fatma nineye
  benzemiyordu (öğretmen perdesi 323 Hz) → yeni tasarım `fatma-5` (231 Hz, hırıltılı; kapı netliği 15) ve bütün
  verisi Kaggle'da yeniden okundu (eskisi `cikti/veri/fatma_eski/`). Kısa cümleler bozuluyordu ("Sınavı geçtim!" →
  "sınava"): `veri/kisa_cumleler.tsv` (245) ve `veri/masal_cumleler.tsv` (38) eklendi, 1 sn'den kısa kayıtlar
  sessizlikle 1 sn'ye tamamlanır. Okunuşlar turkce-vurgu 0.1.1 ile yenilendi. t3'ten 1 tur ince ayar
  (`veri/egitim_vurgu_t6/`, Fatma ×2, kısa ×3, masal ×2 + 5.000 prova kaydı; RTX 4090, ~5 saat):
  `t6-epoch_2nd_00000.pth`, doğrulama kaybı 0,286.

## Ölçümler

12 yeni cümle × 4 ses, nötr ton (`asamaKarsilastir.py`, git geçmişinde; dinleme sayfası `dinleti/asama.html`):

| | 2. aşama | 3. aşama |
|---|---|---|
| Anlaşılırlık (Whisper medium, sözcük benzerliği) | %100 | %100 |
| Doğallık (UTMOS) | 3,58 | 3,60 |

UTMOS İngilizce dinleyicilerle eğitildi; Türkçe vurguyu yargılayamaz, yalnız yapaylığı yakalar.
Vurgunun sınavı kulaktır: kör test iki aşamayı hangisinin hangisi olduğunu göstermeden dinletti
(`korTest.py` ve sayfası git geçmişinde). İlk dinleyici (07.10.2026, 15 soru): fark yok 10,
2. aşama daha doğal 3, 3. aşama daha doğal 1, yanıtsız 1. Açık vurgu işaretleri duyulur bir iyileşme
getirmedi; büyük olasılıkla öğretmen sesler vurguyu zaten doğru okuyordu ve 2. aşama onu sesten öğrenmişti.
Varsayılan yine 3. aşamadır: 9 ton (2. aşamada üzgün, kızgın, fısıltı yok), kalite eşit.

**Duyguların ayırt ediciliği** ([araclar/deney/tonOlc.py](araclar/deney/tonOlc.py), 6 ses × 8 duygu × 6 cümle, 07.10.2026).
Her duygunun nötrden akustik farkı (perde, perde dalgalanması, güç, hız, seslilik) öğretmende ve öğrencide; oran
1 = öğretmen kadar belirgin: fısıltı 1,06 · neşeli 0,91 · kızgın 0,90 · üzgün 0,88 · özür 0,84 · merak 0,80 ·
heyecan 0,78 · uyarı 0,63. Yeni eklenen üç duygu (üzgün, kızgın, fısıltı) ötekilerden zayıf değil. Fark sese göre
değişiyor: Ayşe, Zeynep, Hasan'da duygular öğretmen kadar ya da daha belirgin; Serdar'da neşeli, kızgın ve uyarı
perdede ayrılıyor ama dalgalanma ve hızda öğretmeni izlemiyor; Damla'da uyarı ve merak zayıf.

**Serdar'ın zayıf duyguları neden düzelmiyor** (08.10.2026). İki deneme sonuç vermedi: stil vektörünü öğretmenin
en belirgin kayıtlarından seçmek/itmek (`tonDuzelt.py`, git geçmişinde) doğallığı bozdu; zayıf ses ×
duygular ağırlıklı 1 kısa tur ince ayar ([bulut/kiralik/asama4_odak.py](bulut/kiralik/asama4_odak.py), 3,2 $,
`t4-epoch_2nd_00000.pth`) ortalamayı az artırdı (özür 0,84→0,93, fısıltı 1,06→1,12) ama Serdar'ı kıpırdatmadı ve
nötrde UTMOS 3,03→2,97, Damla'da bir cümle başı hatası ("Kai seri") getirdi → varsayılan t3'te kaldı
(`modelKarsilastir.py`, git geçmişinde; `dinleti/model.html`). Sebep veride: öğretmen (VoxCPM2)
Serdar'ın "kalın, otoriter, sakin" kimlik kaydını tarife baskın tutmuş. Öğretmen kayıtlarında duygunun nötrden
ayrışması (etki büyüklüğü, 25 cümle): neşeli Serdar 0,74 · Hasan 1,58 · Kuzey 1,85 · Damla 2,54; uyarı 1,11 / 1,62 /
1,28 / 2,13. Öğrenci veride olmayanı öğrenemez; çözüm Serdar'ın duygulu cümlelerini öğretmene daha güçlü tarifle
ya da duygulu bir kimlik kaydıyla yeniden ürettirmek. Ayrıca kalın seste pyin perde ölçümü oynak (6 cümlede neşeli
−3, 25 cümlede +1,2 yarım ton): Serdar'ın tonOlc oranları güvenilir değil.

**Serdar yeniden üretildi** (09.10.2026). Güçlü tarif denendi ([ogretmen/serdarDene.py](ogretmen/serdarDene.py),
`dinleti/serdar.html`); neşeli güçlü tarif + cfg 3, uyarı güçlü tarif + cfg 2 seçildi. 468 cümle Kaggle'da ücretsiz
yeniden okundu (o günkü `kaggleIs.py serdar`; bugün `kaggleIs.py yeniden`, 452'si kapıdan geçti; eskiler `cikti/veri/serdar/eski_duygu/`). Öğretmende
nötrden ayrışma: neşeli 1,38 → 4,10, uyarı 1,67 → 3,20. 3. aşamadan 1 kısa tur ince ayar
(o günkü `serdarYeniden.py`, git geçmişinde; bugün [bulut/kiralik/yenile.py](bulut/kiralik/yenile.py), 3,2 $, `t5-epoch_2nd_00000.pth`, doğrulama kaybı
0,290): öğrenci Serdar'ın nötrden perde farkı neşeli +2,7 → +5,1, uyarı +2,0 → +4,6 yarım ton. Öteki seslerde kazanç
yok, Damla'da "Kayseri" bölündü → stüdyoda yalnız Serdar t5 ile konuşur (`studyo.SES_AGIRLIK`).

**Soru ezgisi ve kalan zayıflar** (09.10.2026, [araclar/deney/zayifTeshis.py](araclar/deney/zayifTeshis.py), 25 cümle). Merakın
asıl işareti Türkçe evet/hayır sorusunda soru ekinden önceki sözcüğün vurgulu hecesindeki tepedir ("olaBİLİR misin?",
"ANkara mı?"). Öğretmen yapıyor (tepe Damla +3, Kuzey/Zeynep +4,5 yarım ton), öğrenci düz geçiyordu (Damla'da sonu
düşürüyordu). Çözüm kural: `model.soru_egrisi` turkce-vurgu'nun vurgu işaretinden o heceyi bulur, F0'a `SORU_TEPE` (6)
yarım ton ekler (çözücüde ~175 ms yumuşatılır; ONNX'te `egri` girdisi). Soru sözcüklü sorulara dokunmaz.
[araclar/deney/soruOlc.py](araclar/deney/soruOlc.py), 6 ses × 12 soru: tepe −1,7..+0,7 → +0,6..+2,9; UTMOS en çok −0,08;
anlaşılırlık aynı (`dinleti/soru.html`). Serdar'ın merakı da böylece duyulur (öğretmen verisinde merak yok, 0,36).
Damla'nın uyarısı aslında öğrenilmişti (6 cümlelik tonOlc yanıltmış): perde +1,4 / öğretmen +2,1; uyarı ve merakta stil
nötrden 1,4 kat uzaklaştırılır (`studyo.TON_KAZANC`; UTMOS değişmedi).

**Hız** (i7-12700H, 6 iş parçacığı): RTF ≈ 0,22 (1 sn ses ≈ 0,22 sn); stüdyoda ilk cümle ≈ 0,9 sn.

## Bilinen tuzaklar

- **Cümle başı dolgusu:** model baştaki dolgu işaretine 250–300 ms süre verip bazen orayı sesle
  dolduruyordu ("Kayseri" → "Enkai seri"). `model.BAS_DOLGU` = 4 birim (100 ms) ile sınırlı; 48 sınama
  cümlesinin hepsi kusursuz duyuldu, UTMOS 3,30 → 3,60.
- **Windows güç kısıtlaması:** penceresiz başlatılan süreç 2,7× yavaş çalışıyordu.
  `ogretmen/yukle.sureci_hazirla()` kısıtlamayı kapatır; uzun işleri başlatan her betik bunu çağırır.
- **İş parçacığı sayısı** çağıran iş parçacığına özgüdür (OpenMP). `Ogrenci.hazirla` her çağrıda
  yeniden verir; stüdyo sunucusu seslendirmeyi tek, sabit bir iş parçacığında yapar.
- **ONNX** sesi değiştirmez (log-mel farkı modelin kendi iki çalıştırması arasındakinin ~1/5'i), ama
  kazancı küçüktür (~%10; kısa cümlede ~%27).

## Düzen

```
ogretmen/        VoxCPM2 öğretmen: kadro.py (sesler), duygu.py (ton tarifleri), yukle.py, veriUret.py
                 (+ adayMetin.json), korpusYap.py (derlem), kadroYap.py + tazeleYap.py (yeni ses seçim sayfaları),
                 serdarDene.py (tarif denemesi)
ogrenci/         StyleTTS2 öğrenci: model.py, hizli.py + onnxAktar.py, studyo.py (işaretler, telaffuz, okunuş),
                 ozdenetim.py (cümleyi dinleyip bozuk sözcüğü onarma, öğrenilen onarımlar), kayit.py (öğretmen
                 kaydı ve arka plan işçisi),
                 studyoSunucu.py (sunucu ve API) + site/ (index.html, stil.css, uygulama.js),
                 veri/ (veriHazirla.py, vurguListe.py), dinle.py, acikAktar.py (açık seslen-tts model klasörü),
                 g2p.py (turkce-vurgu'ya yönlendirir)
ogrenci/StyleTTS2/  yl4579/StyleTTS2 kopyası + Utils/ ağırlıkları + Models/LJSpeech/config.yml (gitignore)
araclar/olcu/    ortak ölçü: olcu.py (Whisper, netlik), dogallik.py (UTMOS), kelimeGuven.py, denetim.py (dil modelsiz
                 CTC ile sözcük telaffuz puanı ve fazladan ses), sinav.py (her modelde 16 ses × 273 cümle: veri/sinav/)
araclar/deney/   karşılaştırma ve teşhis: tonOlc.py, zayifTeshis.py, soruOlc.py, zincir.py, yolKarsilastir.py
                 (modellerin ve 1. yolun dinleme sayfası)
bulut/modal/     Modal (aylık 30 $ ücretsiz kredi, yalnız HTTPS): egitModal.py (yükle, hız denemesi, ayrık eğitim, izle, indir)
bulut/kiralik/   RunPod: runpod.py (aç, üret, eğit, izle, kurtar), plbert.py (Türkçe PL-BERT makinesi), plan.py +
                 uret.py (üretim), egitim/ (ayar.py: config_tr.yml, sitecustomize.py, yama.py: ara kayıt + sürdürme),
                 makine/ (makinede çalışan betikler: calistir.sh, egit.sh, asama23.sh, odak.sh kendini toparlar,
                 nobetci.sh takılınca öldürür / durdurur, devam.sh yeniden açılışta sürdürür), asama3_yeni.py
                 (3. aşama), asama4_odak.py (ince ayar, 6 akışlı yükleme), yenile.py (veri yenileme turları t6-t9)
bulut/kaggle/    Kaggle'da ücretsiz GPU: kaggleIs.py; uret.py (öğretmen verisi), tasarim.py + tazele.py (yeni ses
                 adayları), yeniden.py (plana göre yeniden okuma; D kipi = 1. yol, denetimle en iyi okuma),
                 ogretmenDene.py (öğretmen denemeleri)
dinleti/         ölçü betiklerinin dinleme sayfaları (ses/ gitignore)
kullanici/       telaffuz sözlüğü (gitignore)
veri/ model/ cikti/   veri, indirilen modeller, ağırlıklar ve üretilen sesler (gitignore; kisa_cumleler.tsv ve
                      masal_cumleler.tsv depoda)
```
Tek seferlik eski denemeler ve yerini yenisine bırakan betikler (yerel gece üretimi, ilk aday tasarımı, hız
ölçümleri, EVREN planı…) 09.10.2026'da kaldırıldı; 31.10.2026'da eski ağırlıklarla çalışan karşılaştırmalar
(asamaKarsilastir, korTest, modelKarsilastir, tonDuzelt) ve bırakılan 2. yolun betikleri (sesUzayi, havuz, onEgitim)
kaldırıldı. Hepsi git geçmişinde duruyor.

## Kurulum

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install fastapi uvicorn onnxruntime torch torchaudio transformers
:: kardeş projeler: Masaüstü\ses_projeleri\ (GitLab: gitlab.com/mveyseldelibas/turkce-*)
.venv\Scripts\python -m pip install --no-deps -e ..\..\ses_projeleri\turkce-normalize -e ..\..\ses_projeleri\turkce-vurgu ^
    -e ..\..\ses_projeleri\turkce-duygu -e ..\..\ses_projeleri\turkce-uslup
git clone --depth 1 https://github.com/yl4579/StyleTTS2 ogrenci\StyleTTS2
.venv\Scripts\python -c "from huggingface_hub import hf_hub_download as h; h('yl4579/StyleTTS2-LJSpeech', 'Models/LJSpeech/config.yml', local_dir='ogrenci/StyleTTS2')"
```

Çalışma anında `model.Ogrenci` yapıyı `ogrenci/StyleTTS2/Models/LJSpeech/config.yml`'den, ASR/JDC/PL-BERT
iskeletini `Utils/`'ten kurar (bu dosyalar klonla gelir); eğitilmiş ağırlıklar üstüne bizim `.pth` dosyasından
yüklenir. turkce-duygu ve turkce-uslup modelleri kendi `model/` klasörlerinden (yoksa Hugging Face'ten) okunur.
Öğrenci dosyaları (`cikti/ogrenci/t10-*`) ayrıca kopyalanır; depoda değildir.

## Yeniden eğitmek

Kiralık GPU anahtarı `%USERPROFILE%\.runpod\api_key` dosyasında durur (depoda değil).

```
python bulut/kiralik/plan.py                     # öğretmen üretim planı (ses × ton × cümle)
python bulut/kiralik/runpod.py ac --adet 2       # üretim makineleri; baslat / izle / indir / kapat
python ogrenci/veri/veriHazirla.py               # Whisper kapısından geçenler → veri/egitim
python ogrenci/veri/vurguListe.py                # vurgulu okunuş listeleri → veri/egitim_vurgu
python araclar/deney/yolKarsilastir.py           # cikti/teshis/karsilastirma/dinle.html
python bulut/kiralik/asama3_yeni.py hepsi        # eğitim makinesi: aç, kur, yükle, başlat
python bulut/kiralik/runpod.py egit-izle         # tur ağırlıklarını indirir; bitince makineyi siler
python bulut/kiralik/runpod.py egit-kurtar       # kendini durdurmuş makineyi açıp son ağırlığı indirir
python ogrenci/dinle.py cikti/ogrenci/t10-epoch_2nd_00000.pth      # t10-stiller.npz + dinleti/ogrenci.html
python ogrenci/onnxAktar.py cikti/ogrenci/t10-epoch_2nd_00000.pth  # t10-ses.onnx (üç sınavla)
```

Türkçe PL-BERT'in yerel klasörü `TURKCE_PLBERT` ortam değişkeniyle değiştirilebilir (varsayılan
`Masaüstü/ses_projeleri/turkce-plbert`). Kaggle kimliği `%USERPROFILE%\.kaggle\access_token` dosyasında durur.

Var olan veriyi yenileyip kısa ince ayar yapmak (10.10'daki t6 akışı; her adım ayrı çağrılır):

```
python bulut/kiralik/yenile.py plan              # veri/plan_yeniden: yeni sesin bütün cümleleri + kısa/masal cümleleri
python bulut/kaggle/kaggleIs.py yeniden          # Kaggle'da ücretsiz okutma; yeniden-durum / yeniden-indir
python bulut/kiralik/yenile.py birlestir         # indirilenler cikti/veri/'ye
python bulut/kiralik/yenile.py veri              # veriHazirla.py + vurguListe.py
python bulut/kiralik/yenile.py liste             # ağırlıklı liste → veri/egitim_vurgu_t6
python bulut/kiralik/yenile.py egit              # ağ kontrolü (SSH portu kapalıysa kiralamaz), makine, izleme → t6-*.pth
python ogrenci/dinle.py cikti/ogrenci/t6-epoch_2nd_00000.pth       # t6-stiller.npz
python ogrenci/onnxAktar.py cikti/ogrenci/t6-epoch_2nd_00000.pth   # t6-ses.onnx
```

Masraf, takvim ve öğrenci seçiminin gerekçesi: [YOL.md](YOL.md). Seslen kodu MIT ([LICENSE](LICENSE)); kullanılan modellerin ve verilerin lisansları: [LISANSLAR.md](LISANSLAR.md). Ürün sesleri Damla ve Kuzey açık depoda yoktur (`kullanici/ozel_sesler.json` ile eklenir).
