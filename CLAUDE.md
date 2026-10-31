# Çalışma kuralları

Bu kurallar Seslen ve sesle ilgili kardeş projelerin hepsinde aynıdır: seslen, turkce-normalize, turkce-vurgu,
turkce-plbert, turkce-duygu, turkce-uslup, turkce-tts-sinav. Yanıtlar ve ara mesajlar Türkçe.

## 1. Dosya en çok 700 satır

- Bir kod dosyası (`.py`, `.js`, `.ts`, `.html`, `.css`, `.sh`, `.cmd`) 700 satırı aşamaz.
- Bir değişiklik dosyayı 700'ün üstüne çıkaracaksa, aynı değişiklikte dosyayı anlamlı parçalara böl. Örnekler:
  - sunucu ve uç noktalar ayrı,
  - eğitim ve izleme ayrı,
  - sayfa, stil ve betik ayrı.
- Bölerken davranış değişmez. Çağıranları (import, `sys.path`, betik yolları, paketlemeye kopyalanan dosyalar) ve
  belgeleri aynı değişiklikte güncelle.
- Kural dışı olanlar:
  - veri dosyaları (`.json`, `.jsonl`, sözlük `.txt`),
  - ölçü betiklerinin ürettiği dinleme sayfaları,
  - dışarıdan kopyalanmış kod (`ogrenci/StyleTTS2/`).

## 2. Klasörde en çok 10 dosya

- Bir klasörde 10'dan fazla kod ya da belge dosyası birikirse onları işlevine göre alt klasörlere ayır.
  - Örnek: `araclar/olcu/` (ortak ölçüm kütüphanesi) ve `araclar/deney/` (karşılaştırma ve teşhis betikleri).
- Sayılmayanlar: alt klasörler, `__init__.py`, `__pycache__`.
- Kural dışı olanlar:
  - veri ve çıktı klasörleri (`veri/`, `cikti/`, `wavs/`, `model/`, `dinleti/ses/`),
  - testlerin veri dosyaları.
- Taşırken bütün yolları aynı değişiklikte güncelle: içe aktarmalar, `sys.path`, `Path(__file__).parents[n]`,
  belge örnekleri, README bağlantıları.

## 3. Bölme ya da taşımadan sonra

- Bütün `.py` dosyalarını derle (`python -m py_compile`), testleri çalıştır, taşınan betikleri bir kez içe aktararak dene.
- README'deki "Düzen" bölümünü güncelle.
- Kullanılmayan dosya bırakma. Eski denemeleri sil; git geçmişinde kalırlar. Kalan dosyalardaki atıfları
  "git geçmişinde" diye düzelt.
