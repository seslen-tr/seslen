# -*- coding: utf-8 -*-
"""
EĞİTİM YAMASI (18.10) — StyleTTS2 train_finetune.py ağırlığı yalnız tur sonunda kaydediyordu. 17-18.10 gecesi
RunPod makineyi 9.450. adımda (15.700'ün %60'ı) yeniden başlattı; ara kayıt olmadığı için ilerleme kayboldu.
  - her ARA_KAYIT adımda (varsayılan 1500) Models/<KAYIT>/ara.pth (ağırlık + iyileştirici + adım), önce .tmp yazılır
  - ATLA ortam değişkeni: sürdürülen turun ilk ATLA adımı atlanır (veri yüklenir, eğitilmez)
Sürdürme: odak.sh ara.pth varsa DEVAM=…/ara.pth (load_only_params=False) ve ATLA=<adım> ile başlatır.
Makinede bir kez çalışır (işaretli satır varsa dokunmaz):  python yama.py /workspace/StyleTTS2/train_finetune.py
"""
import sys
from pathlib import Path

ISARET = "# seslen: ara kayıt"
yol = Path(sys.argv[1])
s = yol.read_text(encoding="utf-8")
if ISARET in s:
    print("yama zaten var")
    raise SystemExit(0)
dongu = "        for i, batch in enumerate(train_dataloader):\n"
artis = "            iters = iters + 1\n"
assert s.count(dongu) == 1 and s.count(artis) == 1, "train_finetune.py beklenen biçimde değil"
s = s.replace(dongu, f"""        {ISARET}
        import os as _os
        _ATLA = int(_os.environ.get("ATLA", "0")) if epoch == start_epoch else 0
        _ARA = int(_os.environ.get("ARA_KAYIT", "1500"))
{dongu}            if i < _ATLA:
                continue
""")
s = s.replace(artis, f"""{artis}            if _ARA and (i + 1) % _ARA == 0:          {ISARET}
                _gecici = _os.path.join(log_dir, "ara.pth.tmp")
                torch.save({{"net": {{key: model[key].state_dict() for key in model}}, "optimizer": optimizer.state_dict(),
                            "iters": iters, "epoch": epoch, "adim": i + 1}}, _gecici)
                _os.replace(_gecici, _os.path.join(log_dir, "ara.pth"))
                logger.info("ara kayıt: adım %d" % (i + 1))
""")
yol.write_text(s, encoding="utf-8")
print("yama uygulandı")
