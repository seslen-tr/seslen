#!/bin/bash
# NÖBETÇİ (18.10): eğitim takılırsa (train.log 25 dk ilerlemez, süreç duruyor) süreci öldürür; odak.sh ara kayıttan sürdürür.
# odak.sh hiç çalışmıyorsa ve eğitim bitmediyse 15 dk sonra makineyi durdurur (boşta ücret yazmasın).
read KAYIT_AD GRACE < /workspace/EGITIM_BASLADI
LOG=/workspace/StyleTTS2/Models/$KAYIT_AD/train.log
yoksa=0
while [ ! -f /workspace/EGITIM_BITTI ]; do
  sleep 300
  [ -f /workspace/EGITIM_BITTI ] && exit
  if pgrep -f train_finetune > /dev/null; then
    yoksa=0
    if [ -f "$LOG" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOG") )) -gt 1500 ]; then
      echo "== $(date) nöbetçi: train.log 25 dk ilerlemedi, süreç öldürülüyor" >> /workspace/egit.log
      pkill -9 -f train_finetune
    fi
  elif ! pgrep -f odak.sh > /dev/null; then
    yoksa=$((yoksa + 1))
    if [ $yoksa -ge 3 ]; then
      echo "== $(date) nöbetçi: eğitim de odak.sh de yok, makine durduruluyor" >> /workspace/egit.log
      runpodctl stop pod "$(cat /workspace/.pod_id)"; exit
    fi
  fi
done
