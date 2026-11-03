# -*- coding: utf-8 -*-
"""
ÖĞRENCİ MODELİ (StyleTTS2, iSTFTNet) — işlemcide yükle, stil çıkar, konuş.

Çalışma anında difüzyon YOK: her ses × duygu için stil vektörü (256 sayı:
128 akustik + 128 tonlama) eğitim kayıtlarının ortalamasından bir kez çıkarılır
(stiller.npz). Karakter stüdyosu da bu vektörlerle oynar (karıştırma, kaydırma).

    from model import Ogrenci
    o = Ogrenci("cikti/ogrenci/epoch_2nd_00000.pth")
    wav = o.konus("Merhaba!", o.stil_dosyadan("damla", "neseli"))
"""
from __future__ import annotations

import functools
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

KOK = Path(__file__).resolve().parents[1]
ST = KOK / "ogrenci" / "StyleTTS2"
sys.path.insert(0, str(ST))
sys.path.insert(0, str(KOK / "ogrenci"))
# monotonic_align yalnız eğitimde (hizalama); çıkarımda boş modül yeter
for _ad in ("monotonic_align", "monotonic_align.core"):
    sys.modules.setdefault(_ad, types.SimpleNamespace(maximum_path=None, mask_from_lens=None, maximum_path_c=None))
# Resmi StyleTTS2 / kendi ağırlıklarımız eski pickle biçiminde (torch ≥ 2.6 weights_only reddeder)
_torch_load = functools.partial(torch.load, weights_only=False)
ORAN = 24000
BAS_DOLGU = 4          # baştaki dolgu işaretinin en uzun süresi (birim = 25 ms); bkz. hazirla()
# KISA CÜMLE (27.10): tek başına 1-2 sözcüklük cümleye model sonda uydurma hece ekliyor ("Evet." → "evetnı", "Dikkat!" →
# "dikkatsınız"; t10 sınavında "kısa" türü %18). Arkasına taşıyıcı cümle eklenip üretilir, ses hizalamadan bilinen son
# harfte kesilir: 16 ses × 8 kısa cümlede takılan 52 → 10. Sondaki dolguyu sınırlamak (49) ve yalnız kesmek (30) yetmedi.
# Taşıyıcı "h" ile başlar (03.11): "Bunu…" ile dudak sesi kesimden önce sızıyordu ("Katkı." → "katkım"; seslen-tts'te
# 4 ses × 20 kısa cümlede birebir duyulan 49 → 71).
KISA_SOZCUK = 2
TASIYICI = " Hem bunu sana daha önce de söylemiştim."
# CÜMLEYE ÖZGÜ STİL (28.10): sabit ses × duygu vektörü her cümleyi aynı ezgi kalıbıyla okutuyordu. Stil difüzyonu
# (eğitimde var, çalışma anında kullanılmıyordu) cümlenin PL-BERT çıktısından stil önerir; sesin stiliyle karıştırılır:
# akustik yarı %10, tonlama yarısı %40 (StyleTTS2 LibriTTS çıkarımı α, β). 4 ses × 6 cümlede takılan 0 → 0, UTMOS
# 3,13 → 3,25, cümle başına +0,2 sn; kulakla "şimdikinden iyi" (duygu yine öğretmende). α 0,3 / β 0,7 okunuşu bozdu (9/24).
DIF_ALFA, DIF_BETA, DIF_ADIM = 0.1, 0.4, 5
# SORU EZGİSİ (09.10): evet/hayır sorusunda ezginin tepesi soru ekinden (mı/mi/mu/mü + kişi ekleri) önceki sözcüğün
# vurgulu hecesindedir ("yardımcı olaBİLİR misin?", "ANkara mı?"; Göksel & Kerslake §21.2). Öğretmen bunu yapıyor
# (tepe: Damla +3, Kuzey/Zeynep +4,5 yarım ton), öğrenci düz geçiyordu (araclar/deney/zayifTeshis.py). Kural F0'a o hecede
# bu kadar yarım ton ekler; çözücüde ~175 ms ile yumuşatıldığı için ölçülen tepe daha küçük (araclar/deney/soruOlc.py: +6 →
# öğrenci tepesi 0'dan +2..+3'e, UTMOS en çok −0,08, anlaşılırlık aynı). Soru sözcüklü sorulara (ne, neden, kaç…)
# dokunmaz — onlarda ezgi sonda düşer.
SORU_TEPE = 6.0
_SORU_EKI = __import__("re").compile(
    r"^m[ɯiuy](s[ɯiuy]n([ɯiuy]z)?|j[ɯiuy][mz]|d[ɯiuy]ɾ?|jd[ɯiuy](m|n|[ɯiuy]z)?|jmɯʃ|jmiʃ|jmuʃ|jmyʃ)?[?.,!]*$")
_UNLU = set("aeɯiouøyœæɛɪʊ")
# HIZ DENGESİ (03.11): model uzun cümleyi hızlı okuyor (21 sözcükte 7,9 hece/sn, 6 sözcükte 5,3; bir makalede "kimi
# yavaş kimi çok hızlı"). hedef_hece verilince cümlenin ham hızı bunun ±HIZ_BANDI içine çekilir, hiz çarpanı üstüne
# uygulanır (özdenetimin bilerek yavaşlattığı aday yavaş kalır). Hedef: stüdyonun sesin doğal hızı (studyo.dogal_hiz).
HIZ_BANDI, HIZ_EN_COK, HIZ_EN_AZ_HECE = 0.07, 1.35, 6
# YEREL HIZ ve DURAKLAMA (03.11, kulak: "bazı yerlerde çok hızlı okumaya başlıyor", "virgül ve noktaya bazen uyuyor
# bazen uymuyor"): cümle içinde YEREL_HECE heceli bölüm cümle ortalamasının YEREL_KAT katından hızlıysa yavaşlatılır;
# virgülde en az VIRGUL_MS, iki noktada en az IKI_NOKTA_MS.
YEREL_HECE, YEREL_KAT = 4, 1.3
VIRGUL_MS, IKI_NOKTA_MS = 125, 200
# DURAKLAMA SESSİZLİKTİR (03.11, Whisper ile 321 makale cümlesi): noktalamanın süresi uzatılınca model o boşluğu sesle
# dolduruyordu ("İkincisi:" → "ikincisihe", "79 il," → "ile"; baştaki dolguda da aynı, bkz. BAS_DOLGU). Noktalama
# modelde en çok DURAK_KALAN birim kalır, geri kalanı dalgaya sessizlik olarak eklenir; cümle son harften SON_PAY birim
# sonra kesilir (seslen-tts model.sessizlik_ekle ile aynı).
DURAK_KALAN, SON_PAY, GECIS = 2, 2, 120    # birim, birim, örnek (5 ms)
# HARF HARF OKUNAN KISALTMA (03.11, kulak: "TR-CBS-Komut" → "trcbs komut"): model harf adlarını yapıştırıp hızlı
# söylüyordu ("terecebese"); art arda harf adları biraz yavaş ve ayrık okunur
HARF_ADLARI = ("a", "be", "ce", "çe", "de", "e", "fe", "ge", "he", "ı", "i", "je", "ka", "le", "me", "ne", "o", "ö",
               "pe", "re", "se", "şe", "te", "u", "ü", "ve", "ye", "ze", "kü", "iks")
HARF_SOZCUK = ("a", "de", "e", "ne", "o", "ye", "be", "ı", "i", "u", "ü", "ö")   # kendisi de sözcük olanlar
HARF_KAT, HARF_ARA_KAT = 1.3, 2.0


def soru_egrisi(okunus: str, sozluk: dict) -> np.ndarray:
    """Okunuşun (g2p, vurgulu) modele giren her işaretine yarım ton eki: soru ekinden önceki sözcüğün vurgulu hecesine
    SORU_TEPE, gerisine 0. Uzunluk TextCleaner'ın çıktısıyla aynı (sözlükte olmayan karakterler düşer)."""
    ek = np.zeros(len(okunus), np.float32)
    if "?" in okunus:
        sozcukler, i = [], 0
        for s in okunus.split(" "):
            sozcukler.append((i, s))
            i += len(s) + 1
        for j in range(1, len(sozcukler)):
            bas, s = sozcukler[j]
            if not _SORU_EKI.match(s.replace("ˈ", "")) or "?" not in okunus[bas:]:
                continue
            b0, onceki = sozcukler[j - 1]
            if not onceki or onceki[-1] in "?.,!;:":
                continue
            v = onceki.rfind("ˈ")
            if v < 0:               # vurgu işareti yoksa son hece: son ünlü + önündeki ünsüz
                v = max((k for k, c in enumerate(onceki) if c in _UNLU), default=0)
                v -= v > 0 and onceki[v - 1] not in _UNLU
            # hecenin sonu: vurgulu ünlüden sonraki İLK ayrı ünlünün (bitişik ünlü aynı çekirdek) önündeki ünsüz
            k = next((k for k in range(v, len(onceki)) if onceki[k] in _UNLU), len(onceki))
            while k < len(onceki) and (onceki[k] in _UNLU or onceki[k] == "ː"):
                k += 1
            sonraki = next((m for m in range(k, len(onceki)) if onceki[m] in _UNLU), None)
            son = len(onceki) if sonraki is None else max(sonraki - 1, k)
            ek[b0 + v: b0 + son] = SORU_TEPE
    return np.array([e for c, e in zip(okunus, ek) if c in sozluk], np.float32)


def sessizlik_ekle(w: np.ndarray, sure: np.ndarray, duraklar: list[tuple[int, int]], son: int = 0,
                   pay: int = SON_PAY) -> np.ndarray:
    """Dalga: son > 0 ise son harften (işaret sırası son) pay birim sonra kesilir, son birimde söner; duraklar
    noktalamanın başına (önceki sözcüğün bitişine) sessizlik olarak eklenir; sonra eklenince sonraki sözcüğün başı
    iki kez duyuluyordu ("yyetmiş", "rrögar"; seslen-tts model.sessizlik_ekle ile aynı)."""
    sinir = np.cumsum(sure) * 600                           # her işaretin bittiği örnek (birim = 600 örnek)
    w = np.array(w, dtype=np.float32).reshape(-1)
    if son:
        w = w[:min(len(w), int(sinir[son]) + pay * 600)]
        n = min(len(w), 600)
        w[len(w) - n:] *= np.linspace(1, 0, n, dtype=w.dtype)
    parca, bas = [], 0
    for i, birim in duraklar:
        p = int(sinir[i - 1])
        if p >= len(w):
            break
        n = min(GECIS, p - bas, len(w) - p)
        w[p - n:p] *= np.linspace(1, 0, n, dtype=w.dtype)
        w[p:p + n] *= np.linspace(0, 1, n, dtype=w.dtype)
        parca += [w[bas:p], np.zeros(birim * 600, w.dtype)]
        bas = p
    return np.concatenate(parca + [w[bas:]]) if parca else w


class Ogrenci:
    def __init__(self, agirlik: str | Path, is_parcacigi: int = 6, vurgu: bool | None = None, onnx: bool = True):
        """vurgu: okunuşa vurgu işareti (ˈ) koy. None: ağırlığın yanındaki ayardan anla — 3. aşama
        (t3-*.pth, Türkçe PL-BERT) vurgulu listeyle eğitildi. PL-BERT'in yapısı iki dilde aynı;
        eğitilmiş ağırlıkları zaten net["bert"] içinde.
        onnx: çözücü için cikti/ogrenci/<önek>ses.onnx varsa onu kullan (ogrenci/aktar/onnxAktar.py; ~2× hızlı,
        sesi değiştirmediği orada sınanır)."""
        torch.set_num_threads(is_parcacigi)
        self.is_parcacigi = is_parcacigi
        a = Path(agirlik)
        self.onek = a.name.split("epoch_")[0]        # "t3-": 3. aşama; stiller de ayrı dosyada
        if vurgu is None:
            ayar = a.with_name(self.onek + "config_tr.yml")
            # Modal'da eğitilenlerin (t9-) listesi /vol/liste/<tur>/ altında: egitim_vurgu_<tur>'dan kopya, vurgulu.
            # 25.10'a dek bu tanınmadı, t9 ve t10 vurgusuz okunuşla konuştu ("merhebe", "bem", fazla hece).
            metin = ayar.read_text(encoding="utf-8") if ayar.exists() else ""
            vurgu = "egitim_vurgu" in metin or "/vol/liste/" in metin
        self.vurgu = vurgu
        eski = os.getcwd()
        os.chdir(ST)            # StyleTTS2 Utils/ yolları göreli
        torch.load, yedek = _torch_load, torch.load
        try:
            from g2p import g2p
            from models import build_model, load_ASR_models, load_F0_models
            from text_utils import TextCleaner
            from utils import recursive_munch
            from Utils.PLBERT.util import load_plbert
            cfg = yaml.safe_load(open("Models/LJSpeech/config.yml"))
            cfg["model_params"]["multispeaker"] = False      # bulut/kiralik/egitim/ayar.py ile aynı
            self.model = build_model(recursive_munch(cfg["model_params"]),
                                     load_ASR_models("Utils/ASR/epoch_00080.pth", "Utils/ASR/config.yml"),
                                     load_F0_models("Utils/JDC/bst.t7"), load_plbert("Utils/PLBERT/"))
            net = torch.load(str(Path(eski, agirlik) if not Path(agirlik).is_absolute() else agirlik),
                             map_location="cpu")["net"]
        finally:
            torch.load = yedek
            os.chdir(eski)
        for k in self.model:
            if k in net:
                self.model[k].load_state_dict({a.removeprefix("module."): b for a, b in net[k].items()}, strict=False)
            self.model[k].eval()
        # Ağırlık normalizasyonunu katla: çıkarımda her çağrıda yeniden hesaplanıyordu (metin tarafının ~%20'si)
        from torch.nn.utils import remove_weight_norm
        for k in self.model:
            for mod in self.model[k].modules():
                try:
                    remove_weight_norm(mod)
                except (ValueError, AttributeError):
                    pass
        self.g2p, self.temiz = g2p, TextCleaner()
        self.stiller: dict[str, np.ndarray] = {}
        self.oturum = None
        yol = KOK / "cikti" / "ogrenci" / f"{self.onek}ses.onnx"
        if onnx and yol.exists():
            import onnxruntime as ort
            se = ort.SessionOptions()
            se.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            se.intra_op_num_threads = is_parcacigi
            self.oturum = ort.InferenceSession(str(yol), se, providers=["CPUExecutionProvider"])
        self._onnx_girdi = {g.name for g in self.oturum.get_inputs()} if self.oturum is not None else set()

    # ── stil ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def stil(self, x: np.ndarray) -> np.ndarray:
        """24 kHz ses → 256 sayılık stil (akustik 128 + tonlama 128)."""
        # Eğitimdekiyle birebir (StyleTTS2 meldataset.py): torchaudio HTK mel; librosa'nın slaney
        # ölçeği farklı mel verir → stil kayar.
        if not hasattr(self, "_mel"):
            import torchaudio
            self._mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        mel = (torch.log(1e-5 + self._mel(torch.from_numpy(x.astype(np.float32)))) + 4) / 4
        m = mel.unsqueeze(0).unsqueeze(0)
        return torch.cat([self.model.style_encoder(m), self.model.predictor_encoder(m)], dim=1)[0].numpy()

    @torch.no_grad()
    def cumle_stili(self, metin: str, stil: np.ndarray, alfa: float = DIF_ALFA, beta: float = DIF_BETA,
                    okunus: str | None = None) -> np.ndarray:
        """Ses × duygu stilini cümleye özgü difüzyon stiliyle karıştırır (bkz. DIF_ALFA). Tohum metinden: aynı cümle
        hep aynı okunur."""
        import zlib
        if not hasattr(self, "_orneklem"):
            from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
            self._orneklem = DiffusionSampler(self.model.diffusion.diffusion, sampler=ADPM2Sampler(),
                                              sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
                                              clamp=False)
        if torch.get_num_threads() != self.is_parcacigi:
            torch.set_num_threads(self.is_parcacigi)
        okunus = self.g2p(metin, vurgu=self.vurgu) if okunus is None else okunus
        t = torch.LongTensor([[0, *self.temiz(okunus), 0]])
        with torch.random.fork_rng():               # örnekleyicinin içindeki gürültü de bu tohumdan; dışarıyı bozmaz
            torch.manual_seed(zlib.crc32(metin.encode()))
            s = self._orneklem(noise=torch.randn((1, 1, 256)),
                               embedding=self.model.bert(t, attention_mask=torch.ones_like(t)),
                               embedding_scale=1.0, num_steps=DIF_ADIM).squeeze(1)[0].numpy()
        r = np.asarray(stil, np.float32)
        return np.concatenate([alfa * s[:128] + (1 - alfa) * r[:128],
                               beta * s[128:] + (1 - beta) * r[128:]]).astype(np.float32)

    def stil_dosyadan(self, ses: str, duygu: str = "notr") -> np.ndarray:
        if not self.stiller:
            v = np.load(KOK / "cikti" / "ogrenci" / f"{self.onek}stiller.npz")
            self.stiller = {k: v[k] for k in v.files}
        return self.stiller[f"{ses}/{duygu}"]

    # ── konuş ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def hazirla(self, metin: str, stil: np.ndarray, hiz: float = 1.0, virgul_ms: int = 0, okunus: str | None = None,
                uzat: list[tuple[int, int, float]] | None = None, hedef_hece: float | None = None,
                duraklar: list[tuple[int, int]] | None = None):
        """Metin tarafı (sürenin ~%5'i): → (d, t_en, hiza, s, ref, egri) — çözücüye (SesAgi) girdiler.
        virgul_ms: virgül / noktalı virgülün süresine eklenen sessizlik (üslup: masal, talimat).
        okunus: hazır okunuş (verilmezse metinden g2p); uzat: [(baş, son, kat)] işaret aralığının süresi çarpılır
        (özdenetim onarımı, bkz. sozcuk_araliklari). egri: işaret başına perde eki (soru ezgisi, bkz. soru_egrisi).
        duraklar: liste verilirse noktalama modelde en çok DURAK_KALAN birim kalır, duraklamanın geri kalanı
        [(işaret sırası, birim)] olarak buraya eklenir (konus dalgaya sessizlik koyar); verilmezse süre uzatılır."""
        from utils import length_to_mask
        # İş parçacığı sayısı çağıran iş parçacığına özgü (OpenMP): sunucunun istek iş parçacığında yeniden
        # verilmezse 20 çekirdeğin hepsi açılıp ONNX ile çakışıyordu (stüdyoda 2,5× yavaş, 06.10)
        if torch.get_num_threads() != self.is_parcacigi:
            torch.set_num_threads(self.is_parcacigi)
        m = self.model
        okunus = self.g2p(metin, vurgu=self.vurgu) if okunus is None else okunus
        kod = self.temiz(okunus)
        egri = torch.from_numpy(np.concatenate([[0], soru_egrisi(okunus, self.temiz.word_index_dictionary), [0]])
                                .astype(np.float32)).unsqueeze(0)
        t = torch.LongTensor([[0, *kod, 0]])
        n = torch.LongTensor([t.shape[-1]])
        maske = length_to_mask(n)
        ref_s = torch.from_numpy(stil).float().unsqueeze(0)
        s, ref = ref_s[:, 128:], ref_s[:, :128]
        t_en = m.text_encoder(t, n, maske)
        d_en = m.bert_encoder(m.bert(t, attention_mask=(~maske).int())).transpose(-1, -2)
        d = m.predictor.text_encoder(d_en, s, n, maske)
        x, _ = m.predictor.lstm(d)
        ham = torch.sigmoid(m.predictor.duration_proj(x)).sum(axis=-1).squeeze(0)
        hece = sum(c in _UNLU for c in okunus)
        if hedef_hece and hece >= HIZ_EN_AZ_HECE:            # hız dengesi (bkz. HIZ_BANDI)
            oran = hece / max(1e-6, float(ham[1:-1].sum()) * 0.025)
            alt, ust = hedef_hece * (1 - HIZ_BANDI), hedef_hece * (1 + HIZ_BANDI)
            if not alt <= oran <= ust:
                hiz *= float(np.clip((ust if oran > ust else alt) / oran, 1 / HIZ_EN_COK, HIZ_EN_COK))
        ham = (ham / hiz).numpy().astype(np.float64)
        if hedef_hece:                                        # yerel hız (bkz. YEREL_KAT)
            ham = self._yerel_hiz(ham, okunus)
        ham = torch.from_numpy(self._harf_kumesi(ham, okunus)).float()     # harf harf kısaltma (bkz. HARF_KAT)
        sure = ham.round().clamp(min=1).long()
        # Baştaki dolgu işaretine model 10-12 birim (250-300 ms) veriyor ve bazen o boşluğu sesle dolduruyordu
        # ("Kayseri" → "Enkai seri", 06.10). 4 birimde sınırlayınca 48 sınama cümlesinin hepsi kusursuz duyuldu.
        sure[0] = sure[0].clamp(max=BAS_DOLGU)
        # bir süre birimi = 2 mel çerçevesi = 600 örnek = 25 ms. Noktalı virgülde en az 200 ms (25.10: atasözünün
        # ikinci yarısı "vardır; sabreden…" bitişik ve hızlı akıyordu; cümleyi bölmek "vardır"ı bozdu)
        virgul, noktali, iki_nokta = self.temiz(","), self.temiz(";"), self.temiz(":")
        dogal = {}
        for i, k in enumerate(kod, 1):
            if k in noktali or k in virgul or k in iki_nokta:
                dogal[i] = int(sure[i])
            if k in noktali:
                sure[i] += round(max(virgul_ms, 200) / 25)
            elif k in virgul:                         # virgülde en az VIRGUL_MS (03.11: model bazen hiç durmuyordu)
                sure[i] = max(int(sure[i]), round(VIRGUL_MS / 25)) + round(virgul_ms / 25)
            elif k in iki_nokta:
                sure[i] = max(int(sure[i]), round(IKI_NOKTA_MS / 25))
        for bas, son, kat in uzat or ():
            sure[bas:son] = (sure[bas:son].float() * kat).round().clamp(min=1).long()
        if duraklar is not None:                      # duraklamanın fazlası dalgaya sessizlik (bkz. DURAK_KALAN)
            for i, d0 in dogal.items():
                kalan = min(d0, DURAK_KALAN, int(sure[i]))
                if int(sure[i]) > kalan:
                    duraklar.append((i, int(sure[i]) - kalan))
                sure[i] = kalan
        hiza = torch.repeat_interleave(torch.eye(n.item()), sure, dim=1).unsqueeze(0)
        return d, t_en, hiza, s, ref, egri

    def _yerel_hiz(self, sure: np.ndarray, okunus: str) -> np.ndarray:
        """Cümle içinde birden hızlanan bölüm (en az YEREL_HECE heceli ardışık sözcükler) cümlenin sözcüklerdeki
        ortalama hızının YEREL_KAT katını aşıyorsa yavaşlatılır (seslen-tts SesModeli._yerel_hiz ile aynı)."""
        sozluk = self.temiz.word_index_dictionary
        unlu = np.array([0, *[c in _UNLU for c in okunus if c in sozluk], 0], np.float64)
        araliklar = self.sozcuk_araliklari(okunus)
        h = sum(unlu[a:b].sum() for a, b in araliklar)
        d = sum(sure[a:b].sum() for a, b in araliklar)
        if not araliklar or h < HIZ_EN_AZ_HECE or d <= 0 or len(unlu) != len(sure):
            return sure
        sinir = YEREL_KAT * h / d
        sure, bolum = sure.copy(), []
        for a, b in araliklar + [(None, None)]:
            if a is not None:
                bolum.append((a, b))
            if bolum and (a is None or sum(unlu[x:y].sum() for x, y in bolum) >= YEREL_HECE):
                hb, db = sum(unlu[x:y].sum() for x, y in bolum), sum(sure[x:y].sum() for x, y in bolum)
                if hb >= YEREL_HECE and db > 0 and hb / db > sinir:
                    for x, y in bolum:
                        sure[x:y] *= (hb / db) / sinir
                bolum = []
        return sure

    def _harf_kumesi(self, sure: np.ndarray, okunus: str) -> np.ndarray:
        """Harf harf okunan kısaltma ("te re ce be se"): art arda en az iki harf adı, en az biri sözcük olmayan ("ne de"
        kalır) → harfler HARF_KAT, aralarındaki işaretler HARF_ARA_KAT uzar (seslen-tts model.harf_kumesi ile aynı)."""
        yalin = lambda o: "".join(c for c in o if c.isalpha() and c not in "ˈː")       # noqa: E731
        if not hasattr(self, "_harf"):
            self._harf = ({yalin(self.g2p(a, vurgu=self.vurgu)) for a in HARF_ADLARI},
                          {yalin(self.g2p(a, vurgu=self.vurgu)) for a in HARF_SOZCUK})
        harfler, sozcuk = self._harf
        sozluk = self.temiz.word_index_dictionary
        sozcukler = [yalin(p) for p in okunus.split(" ") if any(c.isalpha() for c in p if c in sozluk)]
        araliklar = self.sozcuk_araliklari(okunus)
        if len(sozcukler) != len(araliklar):
            return sure
        harf, sure, i = [s in harfler for s in sozcukler], sure.copy(), 0
        while i < len(harf):
            j = i
            while j < len(harf) and harf[j]:
                j += 1
            if j - i >= 2 and any(sozcukler[k] not in sozcuk for k in range(i, j)):
                for k in range(i, j):
                    a, b = araliklar[k]
                    sure[a:b] *= HARF_KAT
                    if k < j - 1:
                        sure[b:araliklar[k + 1][0]] *= HARF_ARA_KAT
            i = max(j, i + 1)
        return sure

    def sozcuk_araliklari(self, okunus: str) -> list[tuple[int, int]]:
        """Okunuştaki her sözcüğün işaret aralığı [baş, son) (baştaki dolgu 0 sayılır; noktalama dahil değil).
        Sıra normallestir'in sözcükleriyle aynı (g2p sözcük sözcük çevirir)."""
        sozluk = self.temiz.word_index_dictionary
        araliklar, i = [], 1
        for parca in okunus.split(" "):
            bas = son = None
            for c in parca:
                if c not in sozluk:
                    continue
                if c.isalpha():                      # IPA harfleri, ˈ ve ː da (Unicode harf / değiştirici harf)
                    bas = i if bas is None else bas
                    son = i + 1
                i += 1
            i += " " in sozluk                       # boşluk da işaret
            if bas is not None:
                araliklar.append((bas, son))
        return araliklar

    @torch.no_grad()
    def hece_hizi(self, metin: str, stil: np.ndarray) -> float | None:
        """Modelin bu cümleyi hız 1'de kendiliğinden okuyacağı hız (hece/sn; noktalama duraklamaları dahil); kısa
        cümlede (HIZ_EN_AZ_HECE heceden az) None. studyo.dogal_hiz bununla sesin doğal hızını bulur."""
        okunus = self.g2p(metin, vurgu=self.vurgu)
        hece = sum(c in _UNLU for c in okunus)
        if hece < HIZ_EN_AZ_HECE:
            return None
        sure = self.hazirla(metin, stil, okunus=okunus)[2][0].sum(-1)
        return hece / max(1e-6, float(sure[1:-1].sum()) * 0.025)

    def konus(self, metin: str, stil: np.ndarray, hiz: float = 1.0, perde: float = 1.0, tonlama: float = 1.0,
              virgul_ms: int = 0, okunus: str | None = None, uzat: list[tuple[int, int, float]] | None = None,
              ezgi: dict | None = None, hedef_hece: float | None = None) -> np.ndarray:
        """hiz > 1 hızlı; perde: F0 çarpanı (ince-kalın); tonlama: perde dalgalanması çarpanı;
        virgul_ms: virgülde ek duraklama. Son üçü stüdyo kaydırıcıları ve üslup (turkce-uslup) ayarları.
        okunus / uzat: özdenetimin onarımları (bkz. hazirla). ezgi: öğretmenin hece süreleri ve perde çizgisi
        (ezgi.Banka.bul); verilince çözücü PyTorch'ta çalışır (F0'a erişmek için; ~2× yavaş)."""
        from hizli import egri_carpani, tonla
        m = self.model
        okunus = self.g2p(metin, vurgu=self.vurgu) if okunus is None else okunus
        bitis = self._kisa_bitis(metin, okunus)
        if bitis:
            okunus += self._tasiyici_okunus()
        hedef = None if bitis else hedef_hece          # taşıyıcı cümle eklendiyse hece sayısı cümleninki değil
        if ezgi is not None:
            from ezgi import sure_uzat
            hiza0 = self.hazirla(metin, stil, hiz, virgul_ms, okunus, uzat, hedef)[2]
            uzat = list(uzat or []) + sure_uzat(self, okunus, hiza0[0].sum(-1).long().numpy(), ezgi)
        duraklar: list[tuple[int, int]] = []
        d, t_en, hiza, s, ref, egri = self.hazirla(metin, stil, hiz, virgul_ms, okunus, uzat, hedef, duraklar)
        if self.oturum is not None and ezgi is None:
            g = dict(zip(("d", "t_en", "hiza", "s", "ref"), (x.numpy() for x in (d, t_en, hiza, s, ref))))
            g |= {"perde": np.array([perde], np.float32), "tonlama": np.array([tonlama], np.float32)}
            if "egri" in self._onnx_girdi:          # soru ezgisinden önce aktarılmış .onnx'te bu girdi yok
                g["egri"] = egri.numpy()
            w = self.oturum.run(None, g)[0][..., :-50]
        else:
            f0, nn = m.predictor.F0Ntrain(d.transpose(-1, -2) @ hiza, s)
            f0 = f0 * egri_carpani(egri, hiza)
            if ezgi is not None:
                from ezgi import perde_uygula
                f0 = perde_uygula(self, f0, hiza, okunus, ezgi)
            out = m.decoder(t_en @ hiza, tonla(f0, torch.tensor([perde]), torch.tensor([tonlama])), nn, ref)
            w = out.squeeze().numpy()[..., :-50]        # sondaki tık
        # kısa cümlede taşıyıcıdan önce son harften 25 ms sonra, başka cümlede SON_PAY birim sonra kesilir
        return sessizlik_ekle(w, hiza[0].sum(-1).long().numpy(), duraklar, bitis or self._son_harf(okunus),
                              1 if bitis else SON_PAY)

    def _son_harf(self, okunus: str) -> int:
        """Son harfin işaret sırası (baştaki dolgu 0); harf yoksa 0."""
        if not hasattr(self, "_harfsiz"):
            self._harfsiz = {i for c, i in self.temiz.word_index_dictionary.items() if not c.isalpha()}
        kod = self.temiz(okunus)
        son = len(kod)
        while son and kod[son - 1] in self._harfsiz:
            son -= 1
        return son

    def _tasiyici_okunus(self) -> str:
        if not hasattr(self, "_tasiyici"):
            self._tasiyici = " " + self.g2p(TASIYICI.strip(), vurgu=self.vurgu)
        return self._tasiyici

    def _kisa_bitis(self, metin: str, okunus: str) -> int:
        """Kısa cümlede (soru değil) son harfin işaret sırası (baştaki dolgu 0), değilse 0. Bkz. KISA_SOZCUK."""
        metin = metin.strip()
        if len(metin.split()) > KISA_SOZCUK or not metin.endswith((".", "!")):
            return 0
        kod = self.temiz(okunus)
        nokta = set(self.temiz(".!…,"))
        son = len(kod)
        while son > 1 and kod[son - 1] in nokta:
            son -= 1
        return son
