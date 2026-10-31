#!/bin/bash
# ODAK İNCE AYARI (07.10; 18.10 kendini toparlayan): son ağırlıktan 1 tur, ağırlıklı liste (/workspace/egitim_vurgu_odak;
# asama4_odak.py / yenile.py yapar ve gönderir). Difüzyon + SLM açık, LR sabit. Çıktı Models/$KAYIT_AD (Turkce9 → yerelde "t9-").
# 18.10: makine 9.450. adımda yeniden başladı, ara kayıt yoktu, 14 sa boşta ücret yazdı. Şimdi:
#   - egitim/yama.py her 1500 adımda Models/$KAYIT_AD/ara.pth yazar; süreç biterse (çöküş) en çok 4 kez ara.pth'den sürer
#   - nobetci.sh: 25 dk train.log ilerlemezse takılan süreci öldürür (bu döngü sürdürür)
#   - devam.sh: makine yeniden başlarsa (dockerStartCmd) bu betiği yeniden çalıştırır
cd /workspace; exec >> /workspace/egit.log 2>&1
GRACE=${GRACE:-10}
bitir() { echo "== $(date) $1; $GRACE dk sonra durdurulacak"; echo "$1" > /workspace/EGITIM_BITTI; sleep $((GRACE * 60)); runpodctl stop pod "$(cat /workspace/.pod_id)"; exit; }
S=/workspace/StyleTTS2
KAYIT_AD=${KAYIT_AD:-Turkce4}
echo "$KAYIT_AD $GRACE" > /workspace/EGITIM_BASLADI
export PYTHONPATH=/workspace/paket_egit:$PYTHONPATH
rm -f /workspace/EGITIM_BITTI
echo "== $(date) odak ince ayarı başlıyor ($KAYIT_AD)"
sed -i 's/    del new_state_dict\["embeddings.position_ids"\]/    new_state_dict.pop("embeddings.position_ids", None)/' $S/Utils/PLBERT/util.py
grep -q '|' /workspace/egitim_vurgu_odak/OOD_texts.txt || sed -i 's/$/|0/' /workspace/egitim_vurgu_odak/OOD_texts.txt
python /workspace/paket_egit/yama.py $S/train_finetune.py || bitir "HATA: yama"
pgrep -f nobetci.sh > /dev/null || (setsid nohup bash /workspace/nobetci.sh < /dev/null > /dev/null 2>&1 &)
for deneme in 1 2 3 4 5; do
  ARA=$S/Models/$KAYIT_AD/ara.pth
  if [ -f "$ARA" ]; then
    export DEVAM=Models/$KAYIT_AD/ara.pth
    export ATLA=$(python -c "import torch; print(torch.load('$ARA', map_location='cpu', weights_only=False)['adim'])")
    echo "== $(date) ara kayıttan sürdürülüyor: adım $ATLA (deneme $deneme)"
  else
    unset DEVAM ATLA
  fi
  cd /workspace
  BASLANGIC=Models/Turkce3/epoch_2nd_00002.pth KAYIT=Models/$KAYIT_AD PLBERT=Utils/PLBERT_tr/ LISTE=/workspace/egitim_vurgu_odak \
    EPOCH=1 DIFF=0 JOINT=0 BATCH=2 MAXLEN=300 LR=${LR:-5e-5} \
    python /workspace/paket_egit/ayar.py || { GRACE=30; bitir "HATA: ayar"; }
  cd $S
  python train_finetune.py --config_path Configs/config_tr.yml
  kod=$?
  [ $kod -eq 0 ] && [ -f "$S/Models/$KAYIT_AD/epoch_2nd_00000.pth" ] && bitir "eğitim bitti (çıkış 0)"
  echo "== $(date) eğitim süreci bitti (çıkış $kod), ağırlık yok; sürdürülecek"
  sleep 30
done
GRACE=30; bitir "HATA: 5 denemede bitmedi"
