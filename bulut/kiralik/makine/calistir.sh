#!/bin/bash
# Kiralık makinede: kur (bir kez) → üret → bitince GRACE dakika bekle → makineyi durdur.
# runpod.py baslat gönderir (ortamda): SESLER = işçi grupları ";" ile ("ayse,koca;elif;…"),
# EK = uret.py ek seçenekleri, GRACE = dakika.
# İŞÇİLER: aynı kartta birkaç uret.py yan yana. Ölçüm (30.09.2026, 4090): iki süreç
# yan yana koşunca ikisi de tek başınaki hızında → özbağlanımlı çözüm kartı doldurmuyor.
# Durdurma: yerel izleyici (runpod.py izle) indirip silmezse para yanmasın diye.
# Durdurulan makinede /workspace (disk) kalır; veri kaybolmaz.
cd /workspace
exec >> /workspace/calistir.log 2>&1
echo "== $(date) başladı: $SESLER $EK"
if [ ! -f /workspace/.kuruldu ]; then
  # voxcpm bağımlılıksız: gradio (yalnız demo arayüzü) pip'i uvicorn sürümlerinde
  # dakikalarca geri izlemeye sokuyordu. Gerekenler elle; torch ≥ 2.6 (5090 / sm120).
  apt-get install -y -qq ffmpeg > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null)
  pip install -q --no-deps voxcpm==2.0.3 || exit 1
  # torch CUDA 12.8 yapısı: PyPI'nin varsayılanı (CUDA 13) 570 sürücülü makinede "CUDA yok" dedi.
  # -U: görüntüdeki torch 2.4 "zaten kurulu" sayılıp güncellenmiyordu (03.10: voxcpm enable_gqa hatası)
  pip install -q -U torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu128 || exit 1
  pip install -q "transformers>=4.36.2" einops inflect addict wetext \
      "modelscope>=1.22.0" "datasets>=3,<4" huggingface-hub pydantic tqdm simplejson sortedcontainers soundfile \
      librosa matplotlib funasr argbind safetensors faster-whisper==1.2.1 soxr || exit 1
  python -c "from huggingface_hub import snapshot_download as s; s('openbmb/VoxCPM2', local_dir='/workspace/VoxCPM2')" || exit 1
  touch /workspace/.kuruldu
fi
# faster-whisper (ctranslate2) cuDNN / cuBLAS'ı pip'in nvidia paketlerinden bulsun
export LD_LIBRARY_PATH=$(python -c "import nvidia.cublas, nvidia.cudnn, os; print(os.path.join(nvidia.cublas.__path__[0], 'lib') + ':' + os.path.join(nvidia.cudnn.__path__[0], 'lib'))"):$LD_LIBRARY_PATH
mkdir -p /workspace/cikti
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 24 GB kartta 3 işçi (her biri ~7,5 GB: VoxCPM2 bf16 + Whisper); 4'ü taşıyor.
# Çöken işçi (bellek tepesi vb.) kaldığı yerden yeniden başlar (uret.py olcum.jsonl'den sürer).
IFS=';' read -ra GRUPLAR <<< "$SESLER"
for g in "${GRUPLAR[@]}"; do
  ( for t in 1 2 3 4 5 6; do
      python /workspace/paket/uret.py --sesler "$g" $EK && break
      echo "== işçi $g çöktü (deneme $t), 60 sn sonra yeniden"; sleep 60
    done ) >> "/workspace/cikti/isci-${g//,/_}.log" 2>&1 &
  sleep 30      # modeller sırayla yüklensin (bellek tepesi)
done
wait
echo "== $(date) üretim bitti; $GRACE dk sonra durdurulacak"
date > /workspace/cikti/BITTI
sleep $((GRACE * 60))
runpodctl stop pod "$(cat /workspace/.pod_id)"   # runpod.py ac: runpodctl config + .pod_id (SSH oturumu kapsayıcı ortamını görmüyor)
