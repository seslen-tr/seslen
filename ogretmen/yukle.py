# -*- coding: utf-8 -*-
"""
VoxCPM2'yi (öğretmen) bu makinede yükler.

İŞLEMCİDE float32. Modelin config'i bfloat16 diyor; GPU'da doğru, ama AMX'siz
işlemcide (i7-12700H) bf16 matris çarpımı yazılımla öykünülüyor: ilk ölçümde
5 sn ses 129 sn (RTF 26). float32 bellekte 2 kat (≈ 9 GB) ama hızlı.

EcoQoS KAPALI. Windows 11 arka plandaki süreci güç tasarrufuna alıp yavaşlatır
(seslen'de ölçüldü: 3×). Uzun üretimlerde bu süreç hep "arka planda".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
MODEL = KOK / "model" / "VoxCPM2"


def sureci_hazirla() -> None:
    if sys.platform != "win32":
        return
    import ctypes
    import ctypes.wintypes as w
    k = ctypes.windll.kernel32
    k.SetPriorityClass(k.GetCurrentProcess(), 0x8000)          # ABOVE_NORMAL

    class PPT(ctypes.Structure):
        _fields_ = [("Version", w.ULONG), ("ControlMask", w.ULONG), ("StateMask", w.ULONG)]
    h = k.OpenProcess(0x0200 | 0x0400, False, os.getpid())
    s = PPT(1, 0x1, 0)
    k.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s))


def ogretmen(cihaz: str = "cpu"):
    """VoxCPM2 nesnesi. cihaz: cpu | cuda."""
    sureci_hazirla()
    import voxcpm.model.voxcpm2 as v2
    if cihaz == "cpu":
        v2.pick_runtime_dtype = lambda device, dtype: "float32"
    from voxcpm import VoxCPM
    return VoxCPM.from_pretrained(str(MODEL), load_denoiser=False, optimize=False,
                                  device=cihaz, local_files_only=True)
