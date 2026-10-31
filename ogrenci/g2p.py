# -*- coding: utf-8 -*-
r"""
TÜRKÇE G2P — ayrı projeye taşındı (02.10): Masaüstü/ses_projeleri/turkce-vurgu (açık kaynak adayı;
vurgu kuralları + Türkçe PL-BERT orada). Ses Atölyesi onu düzenlenebilir paket olarak kullanır:

    .venv\Scripts\python -m pip install -e ..\..\ses_projeleri\turkce-vurgu

Bu dosya yalnız eski içe aktarmalar (from g2p import g2p) bozulmasın diye var.
1. eğitim (01.10) vurgusuz okunuşla yapıldı: g2p(metin) — vurgu=False varsayılan.
"""
from turkce_vurgu.g2p import NOKTALAMA, SEMBOLLER, g2p, normallestir, ozel_kokler, sozcuk_okunusu  # noqa: F401
