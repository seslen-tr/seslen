#!/bin/bash
# Kiralık makinede StyleTTS2 öğrenci eğitimi. runpod.py egit gönderir.
# Ortam: EPOCH (tur), BATCH, MAXLEN, DEVAM (varsa bu ağırlıktan sürer), GRACE (dk).
# /workspace/egitim/ : veri (wavs/, *_list.txt, OOD_texts.txt) — runpod.py yükler.
# Çıktı: /workspace/StyleTTS2/Models/Turkce/ (epoch_2nd_*.pth, train.log)
# Her çıkışta (hata dahil) GRACE dk sonra makine kendini durdurur: runpod.py ac,
# runpodctl'ye anahtarı ve /workspace/.pod_id'ye kimliği yazar (SSH oturumu
# kapsayıcının ortamını görmüyor — 30.09'da bu yüzden iki makine boşta kaldı).
cd /workspace
exec >> /workspace/egit.log 2>&1
bitir() {
  echo "== $(date) $1; $GRACE dk sonra durdurulacak"
  echo "$1" > /workspace/EGITIM_BITTI     # içerik: "eğitim bitti (çıkış N)" — izleyici yalnız çıkış 0da makineyi siler
  sleep $((GRACE * 60))
  runpodctl stop pod "$(cat /workspace/.pod_id)"
  exit
}
echo "== $(date) başladı: EPOCH=$EPOCH BATCH=$BATCH MAXLEN=$MAXLEN DEVAM=$DEVAM"
rm -f /workspace/EGITIM_BITTI
if [ ! -f /workspace/.egit_kuruldu ]; then
  # torch: kalıbın kendi 2.4.1+cu124 sürümü (değiştirmek riskli: 01.10 bir makinede "CUDA unknown error").
  # transformers 4.46: sonraki sürümler .bin yüklerken torch ≥ 2.6 istiyor (CVE-2025-32434); kalıpta 2.4
  pip install -q munch einops einops-exts pydub nltk "accelerate<1.3" "transformers==4.46.3" soundfile librosa pandas \
      tensorboard matplotlib pyyaml cython || bitir "HATA: paketler kurulamadı"
  pip install -q --no-build-isolation git+https://github.com/resemble-ai/monotonic_align.git || bitir "HATA: monotonic_align"
  [ -d StyleTTS2 ] || git clone -q --depth 1 https://github.com/yl4579/StyleTTS2 || bitir "HATA: git clone"
  python -c "from huggingface_hub import hf_hub_download as h; h('yl4579/StyleTTS2-LJSpeech', 'Models/LJSpeech/epoch_2nd_00100.pth', local_dir='/workspace/StyleTTS2'); h('yl4579/StyleTTS2-LJSpeech', 'Models/LJSpeech/config.yml', local_dir='/workspace/StyleTTS2')" || bitir "HATA: LJSpeech ağırlığı"
  touch /workspace/.egit_kuruldu
fi
# Resmi ağırlıklar eski pickle biçiminde: torch ≥ 2.6 weights_only varsayılanını sitecustomize ile gevşet
export PYTHONPATH=/workspace/paket_egit:$PYTHONPATH
python /workspace/paket_egit/ayar.py || bitir "HATA: ayar"
cd /workspace/StyleTTS2
# expandable_segments KULLANMA: torch 2.4te "CUDA illegal memory access" (01.10, iki deneme çöktü)
python train_finetune.py --config_path Configs/config_tr.yml
kod=$?
# Hata ise 30 dk bekle: izleyici (5 dk'da bir) kaydı alsın, makineye bakılabilsin
[ $kod -ne 0 ] && GRACE=30
bitir "eğitim bitti (çıkış $kod)"
