# -*- coding: utf-8 -*-
"""
EZGİCİ EĞİTİMİ — cikti/ezgici/veri + cikti/ezgi'den ezgiciyi eğitir, cikti/ezgici/ezgici.pt'ye yazar. Küçük olduğu için
CPU'da çalışır; kiralık makine gerekmez. Ayrılan %2 cümlede düzeyin saçılımı örneklenerek izlenir (öğretmene yaklaşmalı).

    python ogrenci/ezgici/egit.py [--adim 30000]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KOK / "ogrenci"), str(KOK / "ogrenci" / "ezgici"), str(KOK / "ogretmen")]
VERI = KOK / "cikti" / "ezgici" / "veri"
BANKA = KOK / "cikti" / "ezgi"
CIKTI = KOK / "cikti" / "ezgici" / "ezgici.pt"
EN_COK_HECE = 60
EN_COK_FARK = 12.0          # yarım ton; üstü perde çıkarıcının hatası (geçişte, kısık seste), kayba girmez


def birlestir() -> None:
    """İşçilerin parçaları (<ses>.<i>.npz) tek dosyaya."""
    for ses in sorted({p.name.split(".")[0] for p in VERI.glob("*.?.npz")}):
        parca = [np.load(p) for p in sorted(VERI.glob(f"{ses}.?.npz"))]
        ofset = [0]
        for p in parca:
            ofset += list(ofset[-1] + p["ogr_ofset"][1:])
        np.savez_compressed(VERI / f"{ses}.npz", ofset=np.array(ofset, np.int64),
                            **{k: np.concatenate([p[k] for p in parca])
                               for k in ("sira", "orta", "ogr_orta", "ogr_sure", "ogr_cizgi")})


def kayitlar() -> tuple[list[dict], list[str], list[str]]:
    """Her cümle: koşul (N, K), hedef (1+N, Y) ölçeksiz, maske. Süre farkı cümle içinde ortalaması sıfır."""
    from ag import BITIS, kosul_kur
    meta_sesler = sorted(p.stem for p in VERI.glob("*.npz") if "." not in p.stem)
    sonuc, tonlar = [], set()
    for sid, ses in enumerate(meta_sesler):
        b, v = np.load(BANKA / f"{ses}.npz"), np.load(VERI / f"{ses}.npz")
        meta = json.loads((BANKA / f"{ses}.json").read_text(encoding="utf-8"))
        for k, j in enumerate(v["sira"]):
            a, s = b["ofset"][j], b["ofset"][j + 1]
            x, y = v["ofset"][k], v["ofset"][k + 1]
            if s - a > EN_COK_HECE or v["orta"][k] <= 0 or v["ogr_orta"][k] <= 0:
                continue
            ogr_sure, ogr_cizgi = v["ogr_sure"][x:y], v["ogr_cizgi"][x:y]
            fark = b["cizgi"][a:s] - ogr_cizgi
            duzey = 12 * np.log2(v["orta"][k] / v["ogr_orta"][k])
            sure = np.log(np.maximum(b["sure"][a:s], 1) / np.maximum(ogr_sure, 1))
            sure -= sure.mean()
            hedef = np.zeros((s - a + 1, 6), np.float32)
            maske = np.zeros_like(hedef)
            hedef[0, 0], maske[0, 0] = duzey, 1
            with np.errstate(invalid="ignore"):
                bilinen = np.abs(fark) <= EN_COK_FARK             # NaN da düşer
            hedef[1:, :5], maske[1:, :5] = np.nan_to_num(fark), bilinen
            hedef[1:, 5], maske[1:, 5] = sure, 1
            tonlar.add(meta[j]["ton"])
            sonuc.append({"kosul": kosul_kur(b["oz"][a:s], ogr_sure, ogr_cizgi), "hedef": hedef, "maske": maske,
                          "ses": sid, "ton": meta[j]["ton"], "bitis": BITIS.index(meta[j]["bitis"]),
                          "duzey": float(duzey), "ogr": float(12 * np.log2(v["ogr_orta"][k])), "gercek": float(12 * np.log2(v["orta"][k]))})
    tonlar = sorted(tonlar)
    for r in sonuc:
        r["ton"] = tonlar.index(r["ton"])
    return sonuc, meta_sesler, tonlar


def yigin(veri: list[dict]):
    import torch
    from ag import BOYUT_K, BOYUT_Y, OLCEK
    n = max(len(r["kosul"]) for r in veri) + 1
    k, y, m = np.zeros((len(veri), n, BOYUT_K), np.float32), np.zeros((len(veri), n, BOYUT_Y), np.float32), \
        np.zeros((len(veri), n, BOYUT_Y), np.float32)
    bos = np.ones((len(veri), n), bool)
    for i, r in enumerate(veri):
        u = len(r["kosul"])
        k[i, 1:u + 1], y[i, :u + 1], m[i, :u + 1], bos[i, :u + 1] = r["kosul"], r["hedef"], r["maske"], False
    return torch.from_numpy(k), torch.from_numpy(y) / OLCEK, torch.from_numpy(m), \
        torch.LongTensor([r["ses"] for r in veri]), torch.LongTensor([r["ton"] for r in veri]), \
        torch.LongTensor([r["bitis"] for r in veri]), torch.from_numpy(bos)


def duzey_olc(ag, ayri: list[dict]) -> dict:
    """Ayrılmış cümlelerde ses içi düzey sapması: öğrenci, örnek, öğretmen (yarım ton; soldaki ikisi öğretmene yaklaşmalı)."""
    import torch
    from ag import OLCEK, ornekle
    ag.eval()
    ornek = []
    for i in range(0, len(ayri), 128):
        p = ayri[i:i + 128]
        k, _, _, s, t, b, bos = yigin(p)
        ornek += (ornekle(ag, k, s, t, b, bos, 7 + i)[:, 0, 0] * OLCEK[0]).tolist()
    ag.train()
    sesler = np.array([r["ses"] for r in ayri])
    ogr, gercek = np.array([r["ogr"] for r in ayri]), np.array([r["gercek"] for r in ayri])
    uretilen = ogr + np.array(ornek)

    def sapma(v):
        return round(float(np.mean([v[sesler == s].std() for s in set(sesler) if (sesler == s).sum() > 5])), 2)
    return {"ogrenci": sapma(ogr), "ornek": sapma(uretilen), "ogretmen": sapma(gercek),
            "r": round(float(np.corrcoef(uretilen, gercek)[0, 1]), 3)}


def main(adim: int = 30000, is_parcacigi: int = 8) -> None:
    import torch
    from ag import Ezgici, akis_kaybi
    from veriUret import uyanik_tut
    from yukle import sureci_hazirla
    sureci_hazirla()
    uyanik_tut()
    torch.set_num_threads(is_parcacigi)
    birlestir()
    veri, sesler, tonlar = kayitlar()
    rs = np.random.default_rng(0)
    rs.shuffle(veri)
    k = len(veri) // 50
    ayri, veri = veri[:k], veri[k:]
    uzunluk = np.array([len(r["kosul"]) for r in veri])
    sira = np.argsort(uzunluk)
    ag = Ezgici(len(sesler), len(tonlar))
    print(f"{len(veri)} cümle, {uzunluk.sum()} hece, {sum(p.numel() for p in ag.parameters())} ağırlık, "
          f"{len(sesler)} ses, {len(tonlar)} ton", flush=True)
    ops = torch.optim.AdamW(ag.parameters(), 3e-4, weight_decay=0.01)
    plan = torch.optim.lr_scheduler.OneCycleLR(ops, 1e-3, total_steps=adim, pct_start=0.05)
    t0, toplam, boy = time.perf_counter(), 0.0, 64
    ag.train()
    for i in range(1, adim + 1):
        bas = int(rs.integers(0, max(1, len(sira) - boy * 4)))           # benzer uzunluk: dolgu az, hız iki kat
        grup = rs.choice(sira[bas:bas + boy * 4], boy, replace=False)
        k_, y, m, s, t, b, bos = yigin([veri[j] for j in grup])
        kayip = akis_kaybi(ag, y, k_, s, t, b, bos, m)
        ops.zero_grad(set_to_none=True)
        kayip.backward()
        torch.nn.utils.clip_grad_norm_(ag.parameters(), 1.0)
        ops.step()
        plan.step()
        toplam += float(kayip)
        if i % 500 == 0:
            print(f"{i}/{adim} kayıp {toplam / 500:.4f} · {(time.perf_counter() - t0) / i * 1000:.0f} ms/adım", flush=True)
            toplam = 0.0
        if i % 5000 == 0 or i == adim:
            print(json.dumps(duzey_olc(ag, ayri[:600]), ensure_ascii=False), flush=True)
            torch.save({"ayar": ag.ayar, "agirlik": ag.state_dict(), "sesler": sesler, "tonlar": tonlar}, CIKTI)


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--adim") + 1]) if "--adim" in sys.argv else 30000)
