# -*- coding: utf-8 -*-
"""
Eğitim ayarı: /workspace/StyleTTS2/Configs/config_tr.yml yazar (egit.sh çağırır).

Başlangıç: StyleTTS2 LJSpeech (iSTFTNet çözücü — işlemcide HiFi-GAN'ın iki katı
hızlı; ogrenci/styleCpuOlc.py — git geçmişinde: RTF 0,28 / 0,55). model_params LJSpeech'ten
AYNEN alınır (ağırlıklar birebir yüklensin). multispeaker: false — stil kodlayıcı
her kaydın tınısını zaten çıkarıyor; bayrak yalnız difüzyon modülünü etkiliyor,
çalışma anında difüzyon yok (ses × duygu stili önceden hesaplanmış vektör).
Ortam: EPOCH, BATCH, MAXLEN, DEVAM (sürdürülecek .pth), DIFF, JOINT (başlangıç
turları; 999 = kapalı — SLM çekişmeli adım pahalı, önce kapalı).
3. aşama (vurgu + Türkçe PL-BERT, 02.10): BASLANGIC (yalnız parametreleri alınacak .pth),
KAYIT (log_dir), PLBERT (Utils/... klasörü), LISTE (train/val/OOD klasörü),
YOKSAY (yüklenmeyecek modüller, virgülle — yeni PL-BERT'in üstüne eskisi yazılmasın: bert).
Odak ince ayarı (07.10): LR (lr ve ft_lr; bert_lr = LR / 10) — varsayılan config_ft'ninki.
"""
import os

import yaml

ST = "/workspace/StyleTTS2"
lj = yaml.safe_load(open(f"{ST}/Models/LJSpeech/config.yml"))
ft = yaml.safe_load(open(f"{ST}/Configs/config_ft.yml"))
devam = os.environ.get("DEVAM", "")

ft["model_params"] = lj["model_params"]
ft["model_params"]["multispeaker"] = False
ft.update({
    "log_dir": os.environ.get("KAYIT", "Models/Turkce"),
    "save_freq": 1,
    "log_interval": 50,
    "epochs": int(os.environ.get("EPOCH", 2)),
    "batch_size": int(os.environ.get("BATCH", 4)),
    "max_len": int(os.environ.get("MAXLEN", 300)),
    "pretrained_model": devam or os.environ.get("BASLANGIC", "Models/LJSpeech/epoch_2nd_00100.pth"),
    "ignore_modules": [m for m in os.environ.get("YOKSAY", "").split(",") if m],
    "second_stage_load_pretrained": True,
    "load_only_params": not bool(devam),
})
if os.environ.get("PLBERT"):
    ft["PLBERT_dir"] = os.environ["PLBERT"]
liste = os.environ.get("LISTE", "/workspace/egitim")
ft["data_params"].update({
    "train_data": f"{liste}/train_list.txt",
    "val_data": f"{liste}/val_list.txt",
    "root_path": "/workspace/egitim/wavs",
    "OOD_data": f"{liste}/OOD_texts.txt",
})
if os.environ.get("LR"):
    lr = float(os.environ["LR"])
    ft["optimizer_params"].update({"lr": lr, "ft_lr": lr, "bert_lr": lr / 10})
ft["loss_params"]["diff_epoch"] = int(os.environ.get("DIFF", 999))
ft["loss_params"]["joint_epoch"] = int(os.environ.get("JOINT", 999))
yaml.safe_dump(ft, open(f"{ST}/Configs/config_tr.yml", "w"), allow_unicode=True, sort_keys=False)
print("config_tr.yml:", {k: ft[k] for k in ("epochs", "batch_size", "max_len", "pretrained_model", "load_only_params",
                                             "log_dir", "PLBERT_dir", "ignore_modules")}, ft["optimizer_params"], ft["data_params"]["train_data"],
      ft["loss_params"]["diff_epoch"], ft["loss_params"]["joint_epoch"], flush=True)
