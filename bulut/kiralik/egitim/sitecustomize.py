# Kiralık makinede PYTHONPATH ile yüklenir (egit.sh).
# 1) StyleTTS2'nin resmi ağırlıkları (yl4579, GitHub + HF) eski pickle biçiminde; torch ≥ 2.6
#    weights_only=True varsayılanı onları reddediyor. Yalnız bu eğitim ortamında gevşetilir.
# 2) cuDNN KAPALI: eğitim 50-800. adımda "illegal memory access" ile çöküyordu; CUDA_LAUNCH_BLOCKING
#    ile asıl hata "GET was unable to find an engine to execute this computation" (cuDNN evrişim
#    geri yayılımı bazı boyutlarda motor bulamıyor) çıktı — torch 2.4 ve 2.6'da aynı (01.10.2026).
import functools
import os

try:
    import torch
    torch.load = functools.partial(torch.load, weights_only=False)
    if os.environ.get("CUDNN", "0") != "1":
        torch.backends.cudnn.enabled = False
except Exception:       # torch yoksa (kurulum sırasında) sessiz geç
    pass
