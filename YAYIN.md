# Açık kaynak yayın listesi

Durum: 11.10.2026. Altı kütüphane ve açık ses modeli **seslen-tts** (6 ses: Ayşe, Zeynep, Emre, Fatma, Hasan, Koca)
yayına hazır; kurum izni var. Kuruluş adı: **seslen-tr** (GitHub ve Hugging Face). Yedi deponun (altı kütüphane +
seslen-tts) GitLab'daki `origin`'inin yanında `github` uzak adresi var (`github.com/seslen-tr/<ad>`) ve dalları oraya
gönderildi; depoların herkese açık olup olmadığı buradan görünmüyor. PyPI ve Hugging Face yüklemeleri bekleniyor.

## 0. Önce

- [x] **Kurum izni.** Projeler iş bilgisayarında, iş saatleri içinde yapıldıysa fikrî haklar kurumda olabilir.
  Yayımlamadan önce yazılı izin alın. Hangi projelerin, hangi lisansla açılacağını izin metnine yazın.
- [ ] **Hesaplar** (kendi girişinizle):
  - [x] GitHub (kuruluş `seslen-tr`),
  - PyPI (iki adımlı doğrulama açık),
  - Hugging Face (yazma anahtarı).
- [ ] **Ürün sesleri açılmaz.** Damla ve Kuzey'in kaydı, stil vektörü ve örneği hiçbir açık depoya girmez.

## 1. Kütüphaneler

Hazır olanlar:

| Proje | Lisans | Pakette | Model (Hugging Face) | Test |
|---|---|---|---|---|
| turkce-normalize (0.1.3) | MIT | kod | — | 81 ✓ |
| turkce-vurgu (0.1.1) | MIT (kod), CC BY-SA 4.0 (Vikisözlük türevi sözlükler) | kod + sözlükler | — | 23 ✓ |
| turkce-tts-sinav | MIT (kod), CC BY-SA 4.0 (veri) | kod + sınav verisi | — | 4 ✓ |
| turkce-duygu | MIT (kod), CC BY 4.0 (model), CC BY-SA 4.0 (`fiiller.txt`) | kod + kurallar | `turkce-duygu` (445 MB) | 6 ✓ |
| turkce-uslup | MIT (kod), CC BY-SA 4.0 (model) | kod + TF-IDF modeli (1,3 MB) | `turkce-uslup` (443 MB, isteğe bağlı) | 6 ✓ |
| turkce-plbert | MIT (kod), CC BY-SA 4.0 (ağırlık) | — (eğitim deposu) | `turkce-plbert` (237 MB) | — |

Ortak hazırlık:
- Testler modelsiz ve kardeş paketler olmadan da geçiyor.
- Her projede şunlar var:
  - GitHub Actions (`.github/workflows/test.yml`; turkce-plbert'te `kontrol.yml`: betik derleme, sözdizimi),
  - `CHANGELOG.md` ve `CITATION.cff`,
  - README başında İngilizce özet,
  - kural dosyası (`CLAUDE.md`).
- Paket adlarının yedisi de PyPI'da boş (09.10.2026).
- Modeller, paket kurulduktan sonra ilk kullanımda Hugging Face'ten kendiliğinden iner.

Yayımlamadan önceki düzeltmeler (09.10'da yapıldı):
- [x] README'lerdeki kardeş proje bağlantıları `github.com/seslen-tr/...` adresleriyle değiştirildi; `pyproject.toml`
      dosyalarında `[project.urls]` var.
- [x] Model deposu varsayılanları `seslen-tr/...` (`HF_DEPO`, ortam değişkeniyle değiştirilebilir:
      `TURKCE_DUYGU_HF`, `TURKCE_USLUP_HF`, `TURKCE_PLBERT_HF`, `SESLEN_TTS_HF`); `hf/yukle.py` dosyaları da aynı
      kuruluşa yükler.

Yayın sırası. Bağımlılıklar isteğe bağlı olduğu için sıra zorunlu değil, ama bu sırayla kurulumlar hep tam olur:

```
# her proje kökünde
pip install build twine && python -m build && twine upload dist/*
turkce-normalize → turkce-vurgu → turkce-tts-sinav → turkce-duygu → turkce-uslup

# modeller (turkce-duygu, turkce-uslup, turkce-plbert): önce gizli depo
huggingface-cli login
python hf/yukle.py            # gizli; Hugging Face'te kartı ve dosyaları kontrol edin
python hf/yukle.py --acik     # herkese açık
```

## 2. Seslen (ses modeli ve uygulama): bütün depo açık (01.11.2026)

11.10'daki "açık çekirdek" kararı (stüdyo kapalı, satılır) değişti: **Seslen deposunun tamamı açık** (MIT), stüdyo,
özdenetim ve öğretmen kaydı dahil. Seslen bağımsız bir projedir; başka projeler (cbs_portal gibi) isterse API'sine
bağlanır, kendi cümle listelerini `kullanici/on_kayit/` altına bırakır.

Açılmayanlar (açık depoda yok):
- **Ürün sesleri Damla ve Kuzey:** tarifleri `kullanici/ozel_sesler.json`'da, kayıtları, stil vektörleri ve
  ağırlıklar `cikti/`'de (ikisi de git dışında). Dosya yoksa Seslen 14 sesle çalışır.
- **Gerçek kişi verisi:** YODAS / ses uzayı eşlemeleri 31.10–01.11'de silindi.
- **Ölçü sayfaları** (`dinleti/`): ürün seslerinin tarifleri geçiyor, sesleri de depoda yok.
- **Geçmiş:** açık depo tek commit'le başlar; bütün geçmiş GitLab'daki özel depoda kalır.

01.11.2026'da açıldı: **github.com/seslen-tr/seslen** (herkese açık, MIT). Yerelde `acik` dalı açık deponun
`main`'idir; özel geçmiş hiçbir zaman oraya gönderilmez (`git push ... main` değil). Güncellerken `main`'in ağacından
`dinleti/` çıkarılıp `acik`'in üstüne tek commit yapılır:

```
set GIT_INDEX_FILE=.git\acik-index && git read-tree main && git rm -r -q --cached dinleti
git commit-tree <git write-tree çıktısı> -p acik -m "..."  →  git update-ref refs/heads/acik <commit>
git push https://github.com/seslen-tr/seslen.git acik:main
```

Açık ses modeli hazır: `Masaüstü/ses_projeleri/seslen-tts`.
- Çıkarım PyTorch'suz, yalnız ONNX Runtime ile. `ogrenci/acikAktar.py` metin tarafını ONNX'e aktardı; Seslen'in
  yoluyla eşdeğerlik birebir, ONNX farkı ≤1,3·10⁻⁵.
- Model klasörü: `cikti/acik/seslen-tts/` (326 MB + `ornekler/`). Yalnız 6 açık ses var; `hf/yukle.py` ürün sesi
  görürse yüklemeyi reddeder. seslen-tts'te yükleme model klasörüyle çağrılır:
  `python hf/yukle.py <model klasörü> [--acik]`.
- 26.10: seslen-tts 0.2.0, `cikti/acik/seslen-tts/` t10 ağırlığından aktarıldı; eski aktarımlar
  `cikti/acik/seslen-tts-t6/` (0.1.0) ve `seslen-tts-t3/`. Paket yoluyla 6 ses × 14 cümle: UTMOS t6'ya göre
  Ayşe 3,73→3,89, Emre 2,42→2,87, Fatma 1,87→2,38, Hasan 2,48→3,26, Koca 3,81→3,89, Zeynep 3,36→3,25; CTC denetiminde
  tek gerçek hata (Hasan "aradığın"). Hugging Face'e gizli yüklendi (etiketler v0.1.0, v0.2.0); GitLab ve GitHub'da
  v0.2.0; 0.2.1 (kısa cümle taşıyıcısı) GitLab ve GitHub'da, model aynı. 01.11: PyPI'da 0.2.1 (turkce-normalize
  0.1.3 ile); temiz ortamda `pip install seslen-tts` PyTorch'suz 139 MB, Zeynep 7,1 sn sesi 2,9 sn'de üretti.
- Temiz ortamda kurulum sınandı: PyTorch yok, ortam 155 MB, komut satırı aracı çalışıyor, Whisper birebir doğru.
- Ses klonlama yok (bilerek).

Yayın sırası (girişlerden sonra):
1. GitHub: 7 depo `seslen-tr` altında (gönderildi; herkese açık değilse depo ayarından açın) + Seslen (01.11, açık).
2. PyPI: turkce-normalize → turkce-vurgu → turkce-tts-sinav → turkce-duygu → turkce-uslup → seslen-tts
   (01.11: hepsi PyPI'da; normalize 0.1.3, seslen-tts 0.2.1 en son yüklenenler).
3. Hugging Face: turkce-duygu, turkce-uslup, turkce-plbert, seslen-tts (önce gizli, kontrol, sonra açık).
