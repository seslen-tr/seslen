# -*- coding: utf-8 -*-
"""
STÜDYO — bir metni bütün ayarlarıyla, cümle cümle seslendirir (GPU'suz, akış hâlinde).

Ayar katmanları (sonraki öncekini ezer):
  1) üslup (turkce-uslup): metnin türüne göre hız / tonlama / duraklama (haber, masal, sohbet…)
  2) ton (turkce-duygu): her cümlenin tonu ve yoğunluğu (merak, uyarı, üzgün…)
  3) kullanıcı: stüdyoda seçilen ayarlar (ses, ton, yoğunluk, perde, hız, tonlama, duraklama, üslup)
  4) metin içi işaretler (en güçlüsü): [üzgün] … [neşeli] …, [duraklama 1.5], [yavaş], [hız 1.2]

Uzun metin paragraf paragraf işlenir (boş satır = paragraf; arası daha uzun duraklama). Telaffuz sözlüğü
(kullanici/telaffuz.json) yalnız modele giden metne uygulanır; ton seçimi özgün metne bakar.
Arayüz: ogrenci/studyoSunucu.py (http://127.0.0.1:8780).

    from studyo import Ayar, Studyo
    st = Studyo()
    for parca in st.seslendir("[neşeli] Günaydın! [duraklama 1] [üzgün] Ama hava yağmurlu.", Ayar(ses="zeynep")):
        oynat(parca.ses)          # her cümle hazır olur olmaz; ardından parca.sessizlik_ms bekle
"""
from __future__ import annotations

import json
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "ogrenci"))
HARF = "a-zA-ZçğıöşüâîûÇĞİÖŞÜÂÎÛ"


@dataclass(frozen=True)
class Ayar:
    ses: str = "zeynep"
    ton: str = "otomatik"            # "otomatik" (turkce-duygu) ya da sabit ton: notr, neseli, … fisilti
    yogunluk: float | None = None    # None: otomatik; 0 → hafif, 1 → tam, 1.5 → abartılı
    perde: float = 1.0               # ince ↔ kalın (F0 çarpanı)
    uslup: str = "otomatik"          # "otomatik" (turkce-uslup) ya da haber, bilgi, talimat, deneme, anlati, masal, sohbet
    hiz: float | None = None         # None: üslubun hızı
    tonlama: float | None = None     # None: üslubun tonlaması
    duraklama_ms: int | None = None  # cümleler arası; None: üslubunki
    virgul_ms: int | None = None     # virgülde ek duraklama; None: üslubunki
    ozdenetim: bool | None = None    # cümleyi dinleyip bozuk sözcüğü onarma (ozdenetim.py); None: SESLEN_OZDENETIM
    difuzyon: bool = True            # cümleye özgü stil (model.cumle_stili); False: sabit ses × duygu stili
    kayit: bool = True               # öğretmen kaydından çal, yoksa sıraya yaz (kayit.py)
    ezgi: str = "tam"                # öğretmenin ezgisi (ezgi.py): "tam" aynı cümle bankadaysa, "benzer" yoksa en
                                     # benzer cümleninki, "kapali" hep öğrencinin kendi ezgisi


@dataclass
class Parca:
    cumle: str
    ses: np.ndarray
    sessizlik_ms: int
    ton: str
    yogunluk: float
    bilgi: dict = field(default_factory=dict)


# Ürün ağırlığı (25.10): t10 — 15 ses 1. yol öğretmen verisiyle (Can G sesi, Elif eski ses), dinlemede onaylandı.
URUN_AGIRLIK = "t10-epoch_2nd_00000.pth"


def son_agirlik() -> Path:
    """Ürün ağırlığı (eski turların ağırlıkları 31.10'da silindi)."""
    return KOK / "cikti" / "ogrenci" / URUN_AGIRLIK


# Sese özel ağırlık. 09.10'dan t10'a dek Serdar t5 ile konuşuyordu (güçlü tarifli duygu verisi); t10'da Serdar da 1. yol
# verisiyle eğitildi ve onaylanan dinleme tek ağırlıkla yapıldı → boş.
SES_AGIRLIK: dict[str, str] = {}
# Ses × ton kazancı (09.10): Damla'nın uyarısı öğretmenin perde yükselişinin ~2/3'ünde kalıyordu (+1,4 / +2,1 yarım
# ton, araclar/deney/zayifTeshis.py). Stil vektörünü nötrden 1,4 kat uzaklaştırmak öğretmene yaklaştırıyor; UTMOS değişmedi
# (1,0: 3,24 · 1,3: 3,26 · 1,5: 3,18), anlaşılırlık aynı. Merakta da aynı (perde +1,5 → +2,0, öğretmen +2,6; soru
# tepesi +1,9 → +2,4; UTMOS 2,68 → 2,66). Duygu şiddeti kaydırıcısı bunun üstüne çarpar.
TON_KAZANC = {("damla", "uyari"): 1.4, ("damla", "merak"): 1.4}


# ── metin içi işaretler ──────────────────────────────────────────────────────────────────────────────
TON_ETIKET = {"nötr": "notr", "notr": "notr", "düz": "notr", "neşeli": "neseli", "neseli": "neseli",
              "sevinçli": "neseli", "heyecan": "heyecan", "heyecanlı": "heyecan", "uyarı": "uyari", "uyari": "uyari",
              "merak": "merak", "meraklı": "merak", "özür": "ozur", "ozur": "ozur", "üzgün": "uzgun", "uzgun": "uzgun",
              "hüzünlü": "uzgun", "kızgın": "kizgin", "kizgin": "kizgin", "öfkeli": "kizgin", "fısıltı": "fisilti",
              "fisilti": "fisilti", "fısıldayarak": "fisilti", "otomatik": "otomatik"}
TON_AD = {"notr": "nötr", "neseli": "neşeli", "heyecan": "heyecan", "uyari": "uyarı", "merak": "merak", "ozur": "özür",
          "uzgun": "üzgün", "kizgin": "kızgın", "fisilti": "fısıltı"}
_ETIKET = re.compile(r"\[([^\]\n]{1,40})\]")
_DURAK = re.compile(r"^(?:duraklama|durak|ara|sessizlik|bekle)(?:\s+([\d.,]+)\s*(ms|sn|s|saniye)?)?$")
_HIZ = re.compile(r"^hız\s+([\d.,]+)$")
_PARAGRAF = re.compile(r"\n[ \t]*\n+")


@dataclass
class Bolum:
    metin: str
    ton: str | None = None          # None: kullanıcının ayarı; "otomatik" ya da sabit ton (işaretten)
    hiz: float | None = None        # işaretten gelen hız (None: ayardaki)
    sonra_ms: int | None = None     # bölümden sonra duraklama (işaret ya da paragraf); None: üslubunki
    paragraf: int = 0


def _kucuk(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def _anahtar(s: str) -> str:
    """Sözlük eşleştirmesi için: I / ı / İ / i aynı sayılır ("GIS", "gis", "Gıs" aynı kayıt)."""
    return s.replace("İ", "i").replace("I", "i").lower().replace("ı", "i")


def etiketleri_coz(metin: str) -> tuple[list[Bolum], list[str]]:
    """Metni işaretlere ve paragraflara göre bölümlere ayırır. Tanınmayan işaretler atılır, uyarı döner."""
    uyari, bolumler = [], []
    ton = hiz = None
    paragraf = 0
    for p_i, par in enumerate(_PARAGRAF.split(metin.strip())):
        if p_i and bolumler:
            if bolumler[-1].sonra_ms is None:
                bolumler[-1].sonra_ms = -1                  # -1: paragraf sonu (üslubun duraklamasının iki katı)
            paragraf += 1
        bas = 0
        parca = []
        for m in _ETIKET.finditer(par):
            parca.append(("metin", par[bas:m.start()]))
            parca.append(("etiket", m.group(1).strip()))
            bas = m.end()
        parca.append(("metin", par[bas:]))
        for tur, deger in parca:
            if tur == "metin":
                if deger.strip():
                    bolumler.append(Bolum(" ".join(deger.split()), ton, hiz, None, paragraf))
                continue
            k = _kucuk(deger)
            if k in TON_ETIKET:
                ton = TON_ETIKET[k]
            elif k in ("yavaş", "yavas"):
                hiz = 0.85
            elif k in ("hızlı", "hizli"):
                hiz = 1.15
            elif k in ("normal", "normal hız", "normal hiz"):
                hiz = None
            elif (m := _HIZ.match(k)):
                hiz = float(np.clip(float(m.group(1).replace(",", ".")), 0.6, 1.6))
            elif (m := _DURAK.match(k)):
                sn = float(m.group(1).replace(",", ".")) if m.group(1) else 0.8
                ms = int(sn if m.group(2) == "ms" else sn * 1000)
                if bolumler:
                    bolumler[-1].sonra_ms = int(np.clip(ms, 0, 10000))
            else:
                uyari.append(f"tanınmayan işaret: [{deger}]")
    return bolumler, uyari


def etiketsiz(metin: str) -> str:
    return " ".join(_ETIKET.sub(" ", metin).split())


# 25.10 dinleme (t10): tek başına okunan 1-2 sözcüklük cümleye model ünlemin / sonun süresinde uydurma hece ekliyor
# ("Dikkat!" → "dikkatsınız", "Merhaba!" tuhaf); sonraki cümleye virgülle bağlanınca 6 seste de temiz ve tonu da doğru
# (turkce-duygu "Dikkat, bu işlem…" → uyarı).
_KISA = re.compile(r"(?:^|(?<=[.!?…]\s))((?:[^\s.!?…;]+\s)?([^\s.!?…;]+))([.!]+)\s+(\w)(?=(\w?))")
_KISALTMA = {"prof", "doç", "dr", "sn", "av", "bkz", "vb", "vs", "örn", "yrd", "mah", "cad", "sok", "no", "op", "uzm"}


def akici(metin: str) -> str:
    def bagla(m: re.Match) -> str:
        son, isaret, ilk, ikinci = m.group(2), m.group(3), m.group(4), m.group(5)
        if isaret == "." and (len(son) < 4 or _kucuk(son) in _KISALTMA or any(c.isdigit() for c in m.group(1))):
            return m.group(0)                  # "Dr. Ahmet", "Madde 3. Sonra" bağlanmaz
        if not ikinci or ikinci.islower():       # "ABD" gibi kısaltmanın büyük harfi kalır
            ilk = _kucuk(ilk)
        return f"{m.group(1)}, {ilk}"
    for _ in range(3):                         # "Evet! Tamam! Gidelim." birden çok kısa cümle
        metin, eski = _KISA.sub(bagla, metin), metin
        if metin == eski:
            break
    return metin


# ── telaffuz sözlüğü ──────────────────────────────────────────────────────────────────────────────────
class Telaffuz:
    """Kullanıcının düzelttiği okunuşlar: {yazılış: okunuş}. Bütün sözcük, büyük-küçük harf duyarsız; ek
    kesme işaretinden sonra korunur (Delibaş'ın → Delibaş okunuşu + 'ın)."""

    def __init__(self, yol: Path = KOK / "kullanici" / "telaffuz.json"):
        self.yol = yol
        self._kilit = threading.Lock()
        self.sozluk: dict[str, str] = json.loads(yol.read_text(encoding="utf-8")) if yol.exists() else {}
        self._derle()

    def _derle(self):
        anahtar = sorted(self.sozluk, key=len, reverse=True)
        self._desen = re.compile(rf"(?<![{HARF}])({'|'.join(map(re.escape, anahtar))})(?![{HARF}])", re.IGNORECASE) \
            if anahtar else None
        self._kucuk = {_anahtar(k): v for k, v in self.sozluk.items()}

    def uygula(self, metin: str) -> str:
        if not self._desen:
            return metin
        return self._desen.sub(lambda m: self._kucuk.get(_anahtar(m.group(1)), m.group(1)), metin)

    def ekle(self, yazilis: str, okunus: str) -> None:
        yazilis, okunus = " ".join(yazilis.split()), " ".join(okunus.split())
        if not yazilis or not okunus or len(yazilis) > 60 or len(okunus) > 120:
            raise ValueError("yazılış ve okunuş boş olamaz (en çok 60 / 120 karakter)")
        with self._kilit:
            self.sozluk[yazilis] = okunus
            self._yaz()

    def sil(self, yazilis: str) -> None:
        with self._kilit:
            self.sozluk.pop(yazilis, None)
            self._yaz()

    def _yaz(self):
        self.yol.parent.mkdir(parents=True, exist_ok=True)
        self.yol.write_text(json.dumps(dict(sorted(self.sozluk.items(), key=lambda x: _kucuk(x[0]))),
                                       ensure_ascii=False, indent=1), encoding="utf-8")
        self._derle()


# ── okunuş önizlemesi ─────────────────────────────────────────────────────────────────────────────────
def okunus(metin: str, telaffuz: Telaffuz | None = None, en_cok: int = 60) -> list[dict]:
    """Metin nasıl okunacak: cümle cümle normalleştirilmiş sözcükler ve vurgulu hecenin konumu.
    [{"cumle", "sozcukler": [{"s": "ankaradan", "v": [0, 2]}, …], "telaffuz": [değişen yazılışlar]}]"""
    from turkce_duygu import cumlelere_bol
    from turkce_vurgu import heceler, normallestir, ozel_kokler, vurgulu_hece
    sonuc = []
    for c in cumlelere_bol(etiketsiz(metin))[:en_cok]:
        degisen = []
        okunacak = c
        if telaffuz and telaffuz._desen:
            degisen = sorted({m.group(1) for m in telaffuz._desen.finditer(c)})
            okunacak = telaffuz.uygula(c)
        kokler = ozel_kokler(okunacak)
        sozcukler = []
        for w in re.findall(r"[a-zçğıöşüâîû]+", normallestir(okunacak)):
            ozel = max((k for k in kokler if w.startswith(k)), key=len, default="")
            k = vurgulu_hece(w, ozel)
            b = heceler(w)
            sozcukler.append({"s": w, "v": None if k is None else [b[k], b[k + 1] if k + 1 < len(b) else len(w)]})
        sonuc.append({"cumle": c, "sozcukler": sozcukler, "telaffuz": degisen})
    return sonuc


class Studyo:
    def __init__(self, agirlik: str | Path | None = None):
        from model import Ogrenci
        from turkce_duygu import Cozumleyici
        self.agirlik = Path(agirlik) if agirlik else son_agirlik()
        self.model = Ogrenci(self.agirlik)
        v = np.load(KOK / "cikti" / "ogrenci" / f"{self.model.onek}stiller.npz")
        self.stiller = {k: v[k] for k in v.files}
        # ses → Ogrenci (SES_AGIRLIK; elle ağırlık verilince yok). O sesin stil vektörleri de o modelinkiler.
        self._ozel: dict = {}
        yuklu: dict = {}
        for ses, ad in ({} if agirlik else SES_AGIRLIK).items():
            yol = KOK / "cikti" / "ogrenci" / ad
            if not yol.exists():
                continue
            if ad not in yuklu:
                yuklu[ad] = Ogrenci(yol)
            m = self._ozel[ses] = yuklu[ad]
            v = np.load(KOK / "cikti" / "ogrenci" / f"{m.onek}stiller.npz")
            self.stiller.update({k: v[k] for k in v.files if k.startswith(f"{ses}/")})
        self._duygu = {}
        self._Cozumleyici = Cozumleyici
        self.telaffuz = Telaffuz()
        self._ozdenetim = None
        from ezgi import Banka
        from kayit import Kayit
        self.kayit = Kayit()
        self.ezgi = Banka()

    def ses_modeli(self, ses: str):
        return self._ozel.get(ses, self.model)

    def ozdenetim(self):
        if self._ozdenetim is None:
            from ozdenetim import Ozdenetim
            self._ozdenetim = Ozdenetim()
        return self._ozdenetim

    @staticmethod
    def ozdenetim_acik(ayar: Ayar) -> bool:
        if ayar.ozdenetim is not None:
            return ayar.ozdenetim
        from ozdenetim import ACIK
        return ACIK

    def tonlar(self, ses: str) -> tuple[str, ...]:
        return tuple(k.split("/")[1] for k in self.stiller if k.startswith(f"{ses}/"))

    def okuma(self, metin: str, ayar: Ayar):
        """Üslup ve okuma ayarı (kullanıcının seçtikleri üslubun varsayılanlarını ezer). Uzun metinde üslup
        ilk ~4000 karakterden bulunur."""
        from turkce_uslup import okuma_ayari, uslup
        if ayar.uslup == "otomatik":
            r = uslup(metin[:4000])
            ad, gerekce = r.uslup, r.gerekce
        else:
            ad, gerekce = ayar.uslup, "kullanıcı"
        degis = {k: getattr(ayar, k) for k in ("hiz", "tonlama", "duraklama_ms", "virgul_ms") if getattr(ayar, k) is not None}
        return ad, okuma_ayari(ad, **degis), gerekce

    def seslendir(self, metin: str, ayar: Ayar = Ayar()):
        """Cümle cümle Parca üretir (ilk cümle hazır olur olmaz oynatılabilir)."""
        mevcut = self.tonlar(ayar.ses)
        if ayar.ses not in self._duygu:
            self._duygu[ayar.ses] = self._Cozumleyici(stiller=mevcut)
        bolumler, uyarilar = etiketleri_coz(metin)
        uslup_ad, ok, uslup_gerekce = self.okuma(etiketsiz(metin), ayar)
        notr = self.stiller[f"{ayar.ses}/notr"]
        for b_i, b in enumerate(bolumler):
            istenen = b.ton if b.ton is not None else ayar.ton
            sonuclar = self._duygu[ayar.ses].metin_cozumle(akici(b.metin))
            # öğretmen kaydı (kayit.py): okuma ayarları elle değiştirilmediyse (öğretmenin kaydı onları izlemez:
            # hız, perde, şiddet, üslup, tonlama, virgül). Ton elle seçildiyse (ayar ya da işaret) yalnız o tonun
            # kaydı çalar. Önce ardışık cümle grupları aranır (hazır kayıtlar birkaç cümle bir arada)
            kayitta = (ayar.kayit and ayar.perde == 1.0 and ayar.hiz is None and b.hiz is None
                       and ayar.yogunluk is None and ayar.uslup == "otomatik" and ayar.tonlama is None
                       and ayar.virgul_ms is None)
            kayit_ton = None if istenen == "otomatik" else istenen
            cumleler = [self.telaffuz.uygula(r.cumle) for r in sonuclar]
            grup: dict[int, tuple[int, np.ndarray]] = {}
            i = 0
            while kayitta and i < len(cumleler):
                for k in range(len(cumleler), i, -1):          # en uzun grup önce, sonra tek cümle
                    w = self.kayit.bul(ayar.ses, " ".join(cumleler[i:k]), kayit_ton)
                    if w is not None:
                        grup[i] = (k, w)
                        i = k
                        break
                else:
                    i += 1
            atla = 0
            for r_i, r in enumerate(sonuclar):
                if r_i < atla:
                    continue
                if istenen == "otomatik":
                    ton, gerekce = r.stil, r.gerekce
                    y = r.yogunluk if ayar.yogunluk is None else ayar.yogunluk
                else:                    # tek tonda (ayar ya da işaret); yoğunluk seçilmediyse tam ton
                    ton = istenen
                    gerekce = f"işaret: [{TON_AD.get(istenen, istenen)}]" if b.ton is not None                         else "bütün metin tek tonda (kullanıcı)"
                    y = 1.0 if ayar.yogunluk is None else ayar.yogunluk
                # Yönergede adımlar düz okunur: turkce-duygu emre hafif uyarı verir (konuşmada doğru), tarifte her
                # adım uyarı gibi çıkıyordu. Gerçek uyarı ve yasak ("Dikkat, sıcak!", "Kabloya dokunmayın.") kalır.
                if (istenen == "otomatik" and uslup_ad == "talimat" and r.eylem not in ("uyari", "yasak")
                        and (r.eylem in ("emir", "rica") or ton == "uyari")):
                    ton, y, gerekce = "notr", 0.0, "talimat üslubunda yönerge adımı düz okunur"
                if ton not in mevcut:
                    ton = "notr"
                s = self.stiller[f"{ayar.ses}/{ton}"]
                if ton != "notr":        # yoğunluk: nötrden tona doğru (stüdyo "duygu şiddeti")
                    # otomatik tonda 0,5 + çözümleyicinin yoğunluğu (hafif duygu da duyulsun); elle ya da sabit
                    # tonda doğrudan çarpan (1 = tam ton)
                    k = 0.5 + y if ayar.yogunluk is None and istenen == "otomatik" else y
                    s = notr + k * TON_KAZANC.get((ayar.ses, ton), 1.0) * (s - notr)
                son, metin_r = r_i, r.cumle
                cumle, model = cumleler[r_i], self.ses_modeli(ayar.ses)
                w = None
                if r_i in grup:                    # öğretmenin kaydı: bir ya da birkaç cümle
                    son, w = grup[r_i][0] - 1, grup[r_i][1]
                    atla = son + 1
                    metin_r = " ".join(x.cumle for x in sonuclar[r_i:son + 1])
                sessizlik = ok.duraklama_ms
                if son == len(sonuclar) - 1 and b.sonra_ms is not None:
                    sessizlik = max(2 * ok.duraklama_ms, 700) if b.sonra_ms == -1 else b.sonra_ms
                kaynak, denetim = ("ogretmen" if w is not None else "ogrenci"), None
                if w is None:
                    if ayar.difuzyon:              # cümleye özgü stil (model.DIF_ALFA)
                        s = model.cumle_stili(cumle, s)
                    ayarlar = {"hiz": b.hiz or ok.hiz, "perde": ayar.perde, "tonlama": ok.tonlama,
                               "virgul_ms": ok.virgul_ms}
                    if ayar.ezgi != "kapali":       # öğretmenin ezgisi (ezgi.py): aynı cümle ya da benzeri
                        ezgi, ezgi_bilgi = self.ezgi.bul(ayar.ses, model.g2p(cumle, vurgu=model.vurgu), ton,
                                                         ayar.ezgi, model.temiz.word_index_dictionary)
                        if ezgi is not None:
                            ayarlar["ezgi"] = ezgi
                            kaynak = "ogrenci+ogretmen_ezgisi"
                    if self.ozdenetim_acik(ayar):
                        w, denetim = self.ozdenetim().uret(model, ayar.ses, cumle, s, notr=notr, **ayarlar)
                    else:
                        w = model.konus(cumle, s, **ayarlar)
                    if kayitta:
                        self.kayit.sira_ekle(ayar.ses, ton, cumle, sabit=kayit_ton is not None)
                bilgi = {"uslup": uslup_ad, "uslup_gerekce": uslup_gerekce, "okuma": ok.__dict__, "eylem": r.eylem,
                         "ton_gerekce": gerekce, "paragraf": b.paragraf, "kaynak": kaynak}
                if denetim:
                    bilgi["ozdenetim"] = denetim
                if kaynak == "ogrenci+ogretmen_ezgisi":
                    bilgi["ezgi"] = ezgi_bilgi
                if b_i == 0 and r_i == 0 and uyarilar:
                    bilgi["uyarilar"] = uyarilar
                yield Parca(metin_r, w, sessizlik, ton, round(float(y), 2), bilgi)

    def tum(self, metin: str, ayar: Ayar = Ayar()) -> tuple[np.ndarray, list[Parca]]:
        from model import ORAN
        parcalar = list(self.seslendir(metin, ayar))
        dalga = []
        for i, p in enumerate(parcalar):
            dalga.append(p.ses / max(1e-6, np.abs(p.ses).max()) * 0.9)
            if i < len(parcalar) - 1:
                dalga.append(np.zeros(int(p.sessizlik_ms / 1000 * ORAN), np.float32))
        return (np.concatenate(dalga) if dalga else np.zeros(0, np.float32)), parcalar
