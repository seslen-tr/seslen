# -*- coding: utf-8 -*-
"""
ÖZDENETİM (27.10) — stüdyo her cümleyi üretince kendisi dinler. Sınav cümleleri her sözcüğü kapsamaz; model
denenmemiş bir sözcükte de bozabilir ("diyarlarda" → "diğarlarda", "sessize" → "sessizi"). Türkçe CTC tanıyıcı
(araclar/olcu/denetim.py) kusur bulursa onarım adayları denenir, en az kusurlu olan seçilir.

Bozuk sözcük: GOP eşiğin (−5) altında, ya da GOP şüpheli (< −1,5) ve tanıyıcı o sözcükte başka harf yazdı (ğ = y:
"diğarlarda" / "diyarlarda", "eyim" / "eğim" Türkçede aynı ses). Başta /
sonda fazladan hece ve okunmayan sözcük de kusur.

Onarımlar (sırayla; ilk kusursuz olanda durulur):
  yavas     bozuk sözcüğün süresi ×1,3            ("diyarlarda" ×1,6'da kusursuz)
  yavas2    ×1,6
  unlu:k    sözcüğün yanlış duyulan k. ünlüsü ×2  ("sessize" → "sessizi", "sabrımı" → "sabrumı": ünlü hatası)
  vurgusuz  sözcükten vurgu işareti kalkar
  durak     sözcükten önce kısa duraklama (virgül)
  notr      cümle nötr tonda (duygulu stil bazı sesleri bozuyor)
  hiz       bütün cümle %10 yavaş

Öğrenme: sözcük onarımı işe yararsa ses × sözcük için kullanici/oto_telaffuz.json'a yazılır, o sözcük sonraki
cümlelerde baştan onarılmış üretilir (yine dinlenir; artık yetmezse silinir). Onarılamayan sözcükler cikti/ozdenetim/
duzelmeyen.jsonl'a eklenir: sonraki eğitimde öğretmenden o sözcüklerle veri üretilir.

Bedel: kusursuz cümlede bir CTC geçişi; kusurlu cümlede aday başına bir üretim + bir geçiş (işlemcide ~2 sn).
SESLEN_OZDENETIM=0 ile kapalı (ya da Ayar(ozdenetim=False)).
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "araclar" / "olcu"))
ACIK = os.environ.get("SESLEN_OZDENETIM", "1") != "0"
ESIK_SUPHE = -1.5            # bu GOP'un altında tanıyıcının harf hatası da bozuk sayılır ("sessizi" −2,3 geçiyordu)
YAVAS = {"yavas": 1.3, "yavas2": 1.6}
UNLU_KAT = 2.0
ONARIMLAR = ("yavas", "yavas2", "unlu", "vurgusuz", "durak", "notr", "hiz")
_HARF = re.compile(r"[a-zçğıöşüâîû]+")
_UNLU = set("aeıioöuüâîû")
_IPA_UNLU = set("aeɯioøuyœæɛ")


def kaynak(metin: str) -> str:
    """Denetimin beklediği metin: sınavla aynı (normallestir, yalnız harfler)."""
    from turkce_vurgu import normallestir
    return " ".join(_HARF.findall(normallestir(metin)))


def _denk(s: str) -> str:
    return s.replace("ğ", "y")


def harf_hatasi(ref: str, duyulan: str) -> list[int]:
    """Sözcük başına harf hatası (ğ = y sayılır)."""
    import denetim
    sozcukler = ref.split()
    sahip = [i for i, s in enumerate(sozcukler) for _ in s]
    hata, ri = [0] * len(sozcukler), 0
    for a, b in denetim.hizala(_denk(ref), _denk(duyulan)):
        if a is None:
            if 0 < ri < len(sahip):
                hata[sahip[ri - 1]] += 1
            continue
        hata[sahip[ri]] += b is None or a != b
        ri += 1
    return hata


def yanlis_unluler(ref: str, duyulan: str) -> list[set[int]]:
    """Her sözcükte tanıyıcının yanlış duyduğu ya da duymadığı ünlülerin sırası (sözcüğün 0., 1. … ünlüsü)."""
    import denetim
    sozcukler = ref.split()
    sonuc: list[set[int]] = [set() for _ in sozcukler]
    harf = [(i, c) for i, s in enumerate(sozcukler) for c in s]            # ref harfi → sözcük
    sira, onceki, k = [], -1, 0
    for i, c in harf:
        k = 0 if i != onceki else k
        sira.append(k if c in _UNLU else None)
        k += c in _UNLU
        onceki = i
    ri = 0
    for a, b in denetim.hizala(_denk(ref), _denk(duyulan)):
        if a is None:
            continue
        if (b is None or a != b) and sira[ri] is not None:
            sonuc[harf[ri][0]].add(sira[ri])
        ri += 1
    return sonuc


class Ozdenetim:
    def __init__(self, sozluk_yolu: Path = KOK / "kullanici" / "oto_telaffuz.json",
                 kayit_yolu: Path = KOK / "cikti" / "ozdenetim" / "duzelmeyen.jsonl"):
        self.sozluk_yolu, self.kayit_yolu = sozluk_yolu, kayit_yolu
        self.ogrenilen: dict[str, dict[str, str]] = (json.loads(sozluk_yolu.read_text(encoding="utf-8"))
                                                     if sozluk_yolu.exists() else {})
        self._kilit = threading.Lock()          # CTC tanıyıcı tek; sunucunun iş parçacıkları sırayla

    # ── puan ──────────────────────────────────────────────────────────────────────────────────────────
    def dinle(self, w: np.ndarray, oran: int, ref: str) -> tuple[float, list[int], dict]:
        """→ (ceza, bozuk sözcüklerin sırası, denetim sonucu). Ceza 0 kusursuz: GOP'un eşik altı kadarı + şüpheli
        sözcükte harf hatası başına 1 + fazladan hece ve okunmayan sözcük 5'er."""
        import denetim
        with self._kilit:
            lp = denetim._logp(np.asarray(w, np.float32).reshape(-1), oran)
            g = denetim.gop(lp, ref)
            d = denetim.puanla(ref, denetim.duy(None, oran, lp), g)
        ceza, bozuk = 0.0, []
        for i, ((s, v), h) in enumerate(zip(g, harf_hatasi(ref, d["duyulan"]))):
            supheli = v < ESIK_SUPHE and h > 0 and len(s) >= 3
            if v < denetim.ESIK_GOP or supheli:
                bozuk.append(i)
            ceza += max(0.0, denetim.ESIK_GOP - v) + (h if supheli else 0)
        ceza += 5 * (d["fazla"][0] >= 2) + 5 * (d["fazla"][1] >= 2) + 5 * len(d["eksik"])
        return round(ceza, 2), bozuk, d

    # ── onarımlar ─────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _isaretler(model, okunus: str) -> list[list[tuple[str, int]]]:
        """Her harfli parça (sözcük) için [(karakter, işaret sırası)] (baştaki dolgu 0)."""
        sozluk = model.temiz.word_index_dictionary
        parcalar, i = [], 1
        for p in okunus.split(" "):
            liste = []
            for c in p:
                if c in sozluk:
                    liste.append((c, i))
                    i += 1
            if any(c.isalpha() for c in p):
                parcalar.append(liste)
            i += " " in sozluk
        return parcalar

    def sozcuk_onar(self, model, okunus: str, degisim: dict[int, str]):
        """Sözcük onarımlarını ({sözcük sırası: onarım}) tek geçişte uygular: önce yazı (vurgu, virgül), sonra son
        okunuştan süre çarpanları. → (okunus, uzat) ya da None (hiçbiri uygulanamadı)."""
        parca = okunus.split(" ")
        sozcuk = [j for j, p in enumerate(parca) if any(c.isalpha() for c in p)]
        uygulanan = {}
        for i, o in degisim.items():
            if i >= len(sozcuk):
                continue
            j = sozcuk[i]
            if o == "vurgusuz":
                if "ˈ" not in parca[j]:
                    continue
                parca[j] = parca[j].replace("ˈ", "")
            elif o == "durak":
                if i == 0 or not parca[sozcuk[i - 1]][-1:].isalpha():
                    continue
                parca[sozcuk[i - 1]] += ","
            uygulanan[i] = o
        if not uygulanan:
            return None
        okunus = " ".join(parca)
        isaret = self._isaretler(model, okunus)
        araliklar = model.sozcuk_araliklari(okunus)
        if len(isaret) != len(sozcuk) or len(araliklar) != len(sozcuk):
            return None
        uzat = []
        for i, o in uygulanan.items():
            if o in YAVAS:
                uzat.append((*araliklar[i], YAVAS[o]))
            elif o.startswith("unlu:"):
                unluler = [t for c, t in isaret[i] if c in _IPA_UNLU]
                for k in map(int, o[5:].split(",")):
                    if k < len(unluler):
                        uzat.append((unluler[k], unluler[k] + 1, UNLU_KAT))
        return okunus, uzat or None

    @staticmethod
    def _cumle_onar(onarim: str, okunus: str, uzat, hiz: float, stil, notr):
        """Cümle düzeyinde onarım (notr, hiz) → (okunus, uzat, stil, hiz) ya da None."""
        if onarim == "notr":
            return None if notr is None or np.allclose(stil, notr) else (okunus, uzat, notr, hiz)
        return okunus, uzat, stil, hiz * 0.9                      # hiz

    # ── üret ──────────────────────────────────────────────────────────────────────────────────────────
    def uret(self, model, ses: str, metin: str, stil, notr=None, hiz: float = 1.0, **kw):
        """Cümleyi üretir, dinler, gerekirse onarır. → (dalga, bilgi)."""
        from model import ORAN
        t0 = time.perf_counter()
        ref = kaynak(metin)
        okunus = model.g2p(metin, vurgu=model.vurgu)
        sozcukler = ref.split()
        eslesir = len(model.sozcuk_araliklari(okunus)) == len(sozcukler)   # sözcük onarımı için sıra eşleşmeli
        bilinen = self.ogrenilen.get(ses, {}) if eslesir else {}
        onceden = {i: bilinen[s] for i, s in enumerate(sozcukler) if s in bilinen}
        temel = (self.sozcuk_onar(model, okunus, onceden) if onceden else None) or (okunus, None)
        w = model.konus(metin, stil, hiz=hiz, okunus=temel[0], uzat=temel[1], **kw)
        ceza, ilk_bozuk, d = self.dinle(w, ORAN, ref)
        bilgi = {"ceza": ceza, "onarim": "ogrenilen" if onceden else None, "duyulan": d["duyulan"]}
        if ceza == 0:
            bilgi["sn"] = round(time.perf_counter() - t0, 2)
            return w, bilgi
        bozuk = ilk_bozuk if eslesir else []
        unlu = yanlis_unluler(ref, d["duyulan"])
        en_iyi = (ceza, w, None, ilk_bozuk, d)
        for onarim in ONARIMLAR:
            if onarim in ("notr", "hiz"):
                a = self._cumle_onar(onarim, temel[0], temel[1], hiz, stil, notr)
                ad = onarim
            else:
                hedef = {i: (f"unlu:{','.join(map(str, sorted(unlu[i])))}" if onarim == "unlu" else onarim)
                         for i in bozuk if onarim != "unlu" or unlu[i]}
                b = self.sozcuk_onar(model, okunus, {**onceden, **hedef}) if hedef else None
                a = None if b is None else (b[0], b[1], stil, hiz)
                ad = hedef
            if a is None:
                continue
            w2 = model.konus(metin, a[2], hiz=a[3], okunus=a[0], uzat=a[1], **kw)
            c2, b2, d2 = self.dinle(w2, ORAN, ref)
            if c2 < en_iyi[0]:
                en_iyi = (c2, w2, ad, b2, d2)
            if c2 == 0:
                break
        c, w, ad, son_bozuk, d = en_iyi
        kalan = {sozcukler[i] for i in son_bozuk}
        self._ogren(ses, ad if isinstance(ad, dict) else {}, sozcukler, ilk_bozuk, kalan, metin)
        bilgi.update({"ceza": c, "ilk_ceza": ceza, "onarim": ad if isinstance(ad, str) or ad is None else
                      sorted(set(ad.values()))[0].split(":")[0], "duyulan": d["duyulan"], "bozuk": sorted(kalan),
                      "sn": round(time.perf_counter() - t0, 2)})
        return w, bilgi

    def _ogren(self, ses: str, hedef: dict[int, str], sozcukler: list[str], ilk_bozuk: list[int], kalan: set[str],
               metin: str) -> None:
        degisti = False
        with self._kilit:
            bilinen = self.ogrenilen.setdefault(ses, {})
            for i in ilk_bozuk:
                s = sozcukler[i]
                if s in kalan:
                    degisti |= bilinen.pop(s, None) is not None        # öğrenilen onarım artık yetmiyor
                elif i in hedef and bilinen.get(s) != hedef[i]:
                    bilinen[s] = hedef[i]
                    degisti = True
            if degisti:
                self.sozluk_yolu.parent.mkdir(parents=True, exist_ok=True)
                self.sozluk_yolu.write_text(json.dumps(self.ogrenilen, ensure_ascii=False, indent=1, sort_keys=True),
                                            encoding="utf-8")
            if kalan:
                self.kayit_yolu.parent.mkdir(parents=True, exist_ok=True)
                with open(self.kayit_yolu, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ses": ses, "cumle": metin, "sozcuk": sorted(kalan),
                                        "zaman": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False) + "\n")
