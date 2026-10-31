#!/bin/bash
# AÇILIŞ KANCASI (18.10): makine yeniden başlarsa (RunPod bakım / donanım) dockerStartCmd bunu çalıştırır. Eğitim
# başlamış ve bitmemişse kurulumu tazeler (kapsayıcı diski silinmiş olabilir) ve odak.sh'yi yeniden başlatır.
[ -f /workspace/EGITIM_BASLADI ] || exit 0
[ -f /workspace/EGITIM_BITTI ] && exit 0
echo "== $(date) makine yeniden başladı, eğitim sürdürülüyor" >> /workspace/egit.log
python -c "import munch, monotonic_align" 2>/dev/null || bash /workspace/kur.sh >> /workspace/egit.log 2>&1
read KAYIT_AD GRACE < /workspace/EGITIM_BASLADI
cd /workspace && GRACE=$GRACE KAYIT_AD=$KAYIT_AD setsid nohup bash odak.sh < /dev/null > /dev/null 2>&1 &
