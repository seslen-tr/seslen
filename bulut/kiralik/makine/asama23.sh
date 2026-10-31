#!/bin/bash
# 2. + 3. AŞAMA, AYNI makinede, arada durmadan (eski asama2.sh'in yerine; 02.10).
#   2. aşama: difüzyon (4. tur) + SLM çekişmeli (5. tur) — vurgusuz, İngilizce PL-BERT.
#   3. aşama: VURGULU okunuş (veri/egitim_vurgu) + TÜRKÇE PL-BERT (Utils/PLBERT_tr), 5. turun
#             ağırlıklarından (yalnız parametre; bert hariç) 3 tur: 0 temel+difüzyon, 1-2 SLM.
#             Girdiler yerelden gönderilir ve /workspace/UC_HAZIR işareti konur; en çok 8 sa beklenir.
# EGITIM_BITTI yalnız en sonda (yerel izleyici onu görünce makineyi siler). 2. aşama sonuçları
# Models/Turkce'de, 3. aşamanınkiler Models/Turkce3'te kalır.
cd /workspace; exec >> /workspace/egit.log 2>&1
GRACE=${GRACE:-10}
bitir() { echo "== $(date) $1; $GRACE dk sonra durdurulacak"; echo "$1" > /workspace/EGITIM_BITTI; sleep $((GRACE * 60)); runpodctl stop pod "$(cat /workspace/.pod_id)"; exit; }
S=/workspace/StyleTTS2
export PYTHONPATH=/workspace/paket_egit:$PYTHONPATH

# ── 2. aşama ────────────────────────────────────────────────────────────────────────────────
while pgrep -f '[t]rain_finetune' > /dev/null; do sleep 60; done
if [ ! -f $S/Models/Turkce/epoch_2nd_00004.pth ]; then
  [ -f $S/Models/Turkce/epoch_2nd_00002.pth ] || { GRACE=30; bitir "HATA: 1. aşamanın 3. tur ağırlığı yok"; }
  echo "== $(date) 2. aşama başlıyor (difüzyon + SLM)"
  # Sürdürmede kayıttaki 'epoch' BİTEN turdur; train_finetune döngüyü ondan başlatıp aynı turu yeniden koşuyor.
  grep -q 'start_epoch += 1' $S/train_finetune.py || \
    sed -i "s/^    n_down = model.text_aligner.n_down/    if not config.get('load_only_params', True): start_epoch += 1  # sesatolye\n&/" $S/train_finetune.py
  grep -q '|' /workspace/egitim/OOD_texts.txt || sed -i 's/$/|0/' /workspace/egitim/OOD_texts.txt
  DEVAM=Models/Turkce/epoch_2nd_00002.pth EPOCH=5 DIFF=3 JOINT=4 BATCH=2 MAXLEN=300 \
    python /workspace/paket_egit/ayar.py || { GRACE=30; bitir "HATA: ayar (2)"; }
  (cd $S && python train_finetune.py --config_path Configs/config_tr.yml)
  kod=$?
  [ $kod -eq 0 ] && [ -f $S/Models/Turkce/epoch_2nd_00004.pth ] || { GRACE=30; bitir "HATA: 2. aşama (çıkış $kod)"; }
fi

# ── 3. aşama ────────────────────────────────────────────────────────────────────────────────
echo "== $(date) 2. aşama bitti; 3. aşama girdileri bekleniyor (UC_HAZIR)"
for i in $(seq 1 480); do [ -f /workspace/UC_HAZIR ] && break; sleep 60; done
[ -f /workspace/UC_HAZIR ] || { GRACE=30; bitir "BEKLEME: 3. aşama girdileri 8 saatte gelmedi (2. aşama tamam)"; }
echo "== $(date) 3. aşama başlıyor (vurgu + Türkçe PL-BERT)"
# Yeni transformers'ta position_ids kayıtta yok; eski util onu silmeye çalışıp KeyError veriyor.
sed -i 's/    del new_state_dict\["embeddings.position_ids"\]/    new_state_dict.pop("embeddings.position_ids", None)/' $S/Utils/PLBERT/util.py
# Başlangıç ağırlıkları yüklenirken 'bert' atlanmalı (yoksa eski İngilizce PL-BERT yeninin üstüne yazılır).
grep -q "ignore_modules=config.get('ignore_modules'" $S/train_finetune.py || \
  sed -i "s/load_only_params=config.get('load_only_params', True))/load_only_params=config.get('load_only_params', True), ignore_modules=config.get('ignore_modules', []))/" $S/train_finetune.py
BASLANGIC=Models/Turkce/epoch_2nd_00004.pth KAYIT=Models/Turkce3 PLBERT=Utils/PLBERT_tr/ LISTE=/workspace/egitim_vurgu \
  YOKSAY=bert EPOCH=3 DIFF=0 JOINT=1 BATCH=2 MAXLEN=300 \
  python /workspace/paket_egit/ayar.py || { GRACE=30; bitir "HATA: ayar (3)"; }
cd $S
python train_finetune.py --config_path Configs/config_tr.yml
kod=$?; [ $kod -ne 0 ] && GRACE=30; bitir "eğitim bitti (çıkış $kod)"
