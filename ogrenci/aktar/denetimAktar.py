# -*- coding: utf-8 -*-
"""
DENETİM TANIYICISI AKTARIMI (03.11.2026) — Seslen'in okunuş denetimindeki Türkçe CTC tanıyıcıyı (wav2vec2 XLS-R 300M,
Common Voice 8 Türkçe; mpoyraz/wav2vec2-xls-r-300m-cv8-turkish, Apache-2.0) seslen-tts'in PyTorch'suz özdenetimi
için ONNX'e aktarır ve 8 bite indirir (1,2 GB → ~0,3 GB). Sınar: PyTorch ile aynı cümlelerde duyulan yazı ve sözcük
başına GOP (araclar/olcu/denetim.py) ne kadar tutuyor.

Çıktı (cikti/acik/seslen-tts/): denetim.onnx, denetim.json (harf sözlüğü, boşluk / dolgu işareti, lisans atfı)

    python ogrenci/aktar/denetimAktar.py [sınama wav klasörü]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "araclar" / "olcu")]
HEDEF = KOK / "cikti" / "acik" / "seslen-tts"
ATIF = ("mpoyraz/wav2vec2-xls-r-300m-cv8-turkish (Apache-2.0), facebook/wav2vec2-xls-r-300m üzerine Mozilla Common "
        "Voice 8.0 Türkçe ile ince ayar; seslen-tts için ONNX'e aktarıldı ve 8 bite indirildi")


def aktar() -> Path:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import denetim
    p, m = denetim._model()
    ham = HEDEF / "denetim_f32.onnx"
    x = torch.zeros(1, 16000 * 3)
    torch.onnx.export(m, (x,), str(ham), input_names=["ses"], output_names=["logits"], opset_version=17,
                      dynamo=False, dynamic_axes={"ses": {1: "T"}, "logits": {1: "F"}}, do_constant_folding=True)
    yol = HEDEF / "denetim.onnx"
    # Yalnız matris çarpımları 8 bit, kanal başına ölçekle; ses kodlayıcının evrişimleri tam duyarlıkta kalır (hepsi
    # 8 bit olunca 30 cümlede duyulan yazı 7'sinde, bayrak 28'inde PyTorch'la aynıydı)
    quantize_dynamic(str(ham), str(yol), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"])
    for f in HEDEF.glob("denetim_f32.onnx*"):
        f.unlink()
    voc = p.tokenizer.get_vocab()
    (HEDEF / "denetim.json").write_text(json.dumps(
        {"sozluk": voc, "bosluk": voc.get("|"), "dolgu": p.tokenizer.pad_token_id, "oran": 16000,
         "olcekle": bool(p.feature_extractor.do_normalize), "atif": ATIF}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return yol


def sina(klasor: Path, adet: int = 30) -> dict:
    """PyTorch ile ONNX (8 bit) aynı mı: duyulan yazı birebir aynı olan oran, GOP bayrağı aynı olan oran."""
    import librosa
    import onnxruntime as ort
    import soundfile as sf
    import denetim
    o = ort.InferenceSession(str(HEDEF / "denetim.onnx"), providers=["CPUExecutionProvider"])
    p, _ = denetim._model()
    dosyalar = sorted(klasor.rglob("*.wav"))[:adet]
    ayni_yazi = ayni_bayrak = 0
    for f in dosyalar:
        x, sr = sf.read(f, dtype="float32")
        lp_t = denetim._logp(x, sr)
        y = librosa.resample(x, orig_sr=sr, target_sr=16000)
        y = (y - y.mean()) / np.sqrt(y.var() + 1e-7)
        lg = o.run(None, {"ses": y[None].astype(np.float32)})[0][0]
        lp_o = lg - np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lg.max(-1, keepdims=True)
        yt, yo = denetim.duy(None, 16000, lp_t), denetim.duy(None, 16000, lp_o)
        ayni_yazi += yt == yo
        ref = yt
        bt = [v < denetim.ESIK_GOP for _, v in denetim.gop(lp_t, ref)]
        bo = [v < denetim.ESIK_GOP for _, v in denetim.gop(lp_o, ref)]
        ayni_bayrak += bt == bo
    return {"dosya": len(dosyalar), "ayni_yazi": ayni_yazi, "ayni_bayrak": ayni_bayrak,
            "boyut_mb": round((HEDEF / "denetim.onnx").stat().st_size / 1e6)}


if __name__ == "__main__":
    yol = aktar() if not (HEDEF / "denetim.onnx").exists() or "--yeniden" in sys.argv else HEDEF / "denetim.onnx"
    k = next((Path(a) for a in sys.argv[1:] if not a.startswith("--")), KOK / "cikti" / "bolme")
    print(yol, sina(k))
