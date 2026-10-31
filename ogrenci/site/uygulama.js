// Seslen stüdyo sitesi: davranış (ayarlar, seslendirme akışı, sesler, telaffuz, API sayfası). Sayfa: index.html
"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const yerel = { al(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
                koy(k, v) { try { localStorage.setItem(k, v); } catch (_) {} } };
const buyuk = (s) => s ? s[0].toLocaleUpperCase("tr") + s.slice(1) : s;
const IKON = {
  oynat: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>',
  durdur: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>',
  ok: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 6l6 6-6 6"/></svg>',
  dinle: '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>',
  sil: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
};
const TUR_RENK = { "yetişkin": "--a-yetiskin", "genç": "--a-genc", "yaşlı": "--a-yasli", "çocuk": "--a-cocuk", "çizgi": "--a-cizgi" };
const TUR_AD = { "yetişkin": "Yetişkin", "genç": "Genç", "yaşlı": "Yaşlı", "çocuk": "Çocuk", "çizgi": "Çizgi film" };
const ORNEK = {
  "Masal": "Bir varmış bir yokmuş, evvel zaman içinde küçük bir köyde Pıtırcık adında meraklı bir sincap yaşarmış. Bir sabah ormanda kocaman bir kutu bulmuş. Acaba içinde ne vardı? Dikkatle yaklaşmış ve kapağı açmış. İçinden rengârenk kelebekler uçuşmuş!",
  "Haber": "Meteoroloji Genel Müdürlüğü, yarın İç Anadolu'da kuvvetli yağış beklendiğini açıkladı. Ankara, Konya ve Eskişehir'de sel riskine karşı sürücülerin dikkatli olması istendi.",
  "Tarif": "Tencereye iki su bardağı su ekleyin. Pirinci yıkayıp suya katın. Kısık ateşte yirmi dakika pişirin. Dikkat, tencere çok sıcak olacak!",
  "İşaretli": "[neşeli] Müjde! Sınavı kazandım. [duraklama 1] [üzgün] Ama en yakın arkadaşım kazanamadı.\n\n[fısıltı] Ona henüz söylemedim. [otomatik] Sence ne yapmalıyım?",
};
const ISARET_AD = { notr: "nötr", neseli: "neşeli", heyecan: "heyecan", uyari: "uyarı", merak: "merak", ozur: "özür", uzgun: "üzgün", kizgin: "kızgın", fisilti: "fısıltı" };

let BILGI = null;
const VARSAYILAN = { ses: "zeynep", ton: "otomatik", yogunluk: null, perde: 1, uslup: "otomatik", hiz: null, tonlama: null, duraklama_ms: null, virgul_ms: null };
let A = { ...VARSAYILAN };

// ── tema ────────────────────────────────────────────────────────
function temaUygula(t) {
  if (t === "acik") document.documentElement.dataset.theme = "light";
  else if (t === "koyu") document.documentElement.dataset.theme = "dark";
  else delete document.documentElement.dataset.theme;
  document.querySelectorAll(".tema button").forEach((b) => b.classList.toggle("etkin", b.dataset.tema === t));
  yerel.koy("sa.tema", t);
}
document.querySelectorAll(".tema button").forEach((b) => b.onclick = () => temaUygula(b.dataset.tema));
temaUygula(yerel.al("sa.tema") || "sistem");

// ── yönlendirme ─────────────────────────────────────────────────
const GORUNUMLER = ["studyo", "sesler", "telaffuz", "api"];
function git() {
  let g = (location.hash || "#studyo").slice(1);
  if (!GORUNUMLER.includes(g)) {             // sayfa içi bağlantı (#isaretRehberi): stüdyoda kal, oraya kaydır
    const hedef = document.getElementById(g);
    g = "studyo";
    if (hedef) setTimeout(() => hedef.scrollIntoView({ behavior: "smooth", block: "start" }), 30);
  }
  document.querySelectorAll(".gorunum").forEach((s) => s.classList.toggle("etkin", s.id === "g-" + g));
  document.querySelectorAll(".menu a").forEach((a) => a.classList.toggle("etkin", a.dataset.gorunum === g));
  if (g === "sesler" && BILGI) sesIzgarasi($("sesIzgara"), false);
}
window.addEventListener("hashchange", git);

// ── ses görselleri ──────────────────────────────────────────────
const sesBul = (ad) => BILGI.sesler.find((s) => s.ad === ad);
function avatar(s, boy = 40) {
  return `<span class="avatar" style="background:var(${TUR_RENK[s.tur] || "--a-yetiskin"});width:${boy}px;height:${boy}px;font-size:${boy * .38}px">${esc(s.gorunen[0])}</span>`;
}
const tonRozet = (t, y) => `<span class="rozet ton-rozet"><span class="nokta" style="background:var(--t-${t})"></span>${esc(buyuk(BILGI.tonlar[t] || t))}${y !== undefined && t !== "notr" ? ` · %${Math.round(y * 100)}` : ""}</span>`;

let ornekSes = null;
function ornekCal(ad, dugme) {
  if (ornekSes) { ornekSes.pause(); ornekSes = null; }
  const eski = dugme.innerHTML;
  dugme.innerHTML = "Hazırlanıyor…"; dugme.disabled = true;
  ornekSes = new Audio(`/api/ornek/${ad}`);
  ornekSes.oncanplay = () => { dugme.innerHTML = "Çalıyor…"; };
  ornekSes.onended = ornekSes.onerror = () => { dugme.innerHTML = eski; dugme.disabled = false; };
  ornekSes.play().catch(() => { dugme.innerHTML = eski; dugme.disabled = false; });
}

function sesSec(ad) { A.ses = ad; ayarCiz(); perdeKapat(); if (location.hash !== "#studyo") location.hash = "#studyo"; }

function sesIzgarasi(hedef, secimde) {
  const gruplar = {};
  for (const s of BILGI.sesler) (gruplar[s.tur] = gruplar[s.tur] || []).push(s);
  hedef.innerHTML = Object.entries(gruplar).map(([tur, liste]) => `
    <div class="grup-baslik">${esc(TUR_AD[tur] || tur)}</div>
    <div class="ses-izgara">${liste.map((s) => `
      <div class="kart ses-kart${s.ad === A.ses ? " secili" : ""}">
        <div class="ust">${avatar(s)}<div><div class="ad">${esc(s.gorunen)}</div><div class="aciklama" style="margin:0">${s.tonlar.length} duygu</div></div></div>
        <div class="ac">${esc(s.aciklama)}</div>
        <div class="eylem"><button class="dugme kucuk" data-dinle="${s.ad}">${IKON.dinle} Dinle</button>
          <button class="dugme kucuk ana" data-kullan="${s.ad}">${secimde ? "Seç" : "Kullan"}</button></div>
      </div>`).join("")}</div>`).join("");
  hedef.querySelectorAll("[data-dinle]").forEach((b) => b.onclick = () => ornekCal(b.dataset.dinle, b));
  hedef.querySelectorAll("[data-kullan]").forEach((b) => b.onclick = () => sesSec(b.dataset.kullan));
}

// ── ayar paneli ─────────────────────────────────────────────────
const KAYDIRICI = [
  { id: "hiz", ad: "Hız", min: 0.6, max: 1.6, adim: 0.01, temel: true, uc: ["Yavaş", "Hızlı"], oto: true, goster: (v) => "×" + v.toFixed(2) },
  { id: "perde", ad: "Ses perdesi", min: 0.7, max: 1.4, adim: 0.01, temel: true, uc: ["Kalın", "İnce"], oto: false, goster: (v) => "×" + v.toFixed(2) },
  { id: "yogunluk", ad: "Duygu şiddeti", min: 0, max: 1.5, adim: 0.05, temel: true, uc: ["Hafif", "Güçlü"], oto: true, goster: (v) => "%" + Math.round(v * 100) },
  { id: "tonlama", ad: "Tonlama", min: 0, max: 2, adim: 0.05, temel: false, uc: ["Düz", "Hareketli"], oto: true, goster: (v) => "×" + v.toFixed(2) },
  { id: "duraklama_ms", ad: "Cümleler arası duraklama", min: 0, max: 2000, adim: 25, temel: false, uc: ["Yok", "2 sn"], oto: true, goster: (v) => (v / 1000).toFixed(2) + " sn" },
  { id: "virgul_ms", ad: "Virgülde ek duraklama", min: 0, max: 800, adim: 25, temel: false, uc: ["Yok", "0,8 sn"], oto: true, goster: (v) => (v / 1000).toFixed(2) + " sn" },
];
function otoDeger(k) {
  if (k.id === "yogunluk") return A.ton === "otomatik" ? null : 1;
  if (A.uslup === "otomatik" || !BILGI) return null;
  return BILGI.usluplar[A.uslup][k.id];
}
function kaydiriciKur() {
  for (const k of KAYDIRICI) {
    const d = document.createElement("div");
    d.className = "kaydirici";
    d.innerHTML = `<div class="k-ust"><span class="ad">${k.ad}</span>${k.oto ? `<button class="oto" id="${k.id}Oto" title="Otomatik: üsluba / metne göre">Otomatik</button>` : ""}
      <span class="deger" id="${k.id}Deger"></span></div>
      <input type="range" id="${k.id}" min="${k.min}" max="${k.max}" step="${k.adim}" aria-label="${k.ad}">
      <div class="uclar"><span>${k.uc[0]}</span><span>${k.uc[1]}</span></div>`;
    (k.temel ? $("temelKaydiricilar") : $("gelismisKaydiricilar")).appendChild(d);
    const g = $(k.id);
    g.addEventListener("input", () => { A[k.id] = Number(g.value); kaydiriciCiz(k); yerel.koy("sa.ayar", JSON.stringify(A)); });
    g.addEventListener("dblclick", () => { A[k.id] = k.oto ? null : 1; kaydiriciCiz(k); });
    if (k.oto) $(k.id + "Oto").onclick = () => { A[k.id] = A[k.id] === null ? Number(g.value) : null; kaydiriciCiz(k); };
  }
}
function kaydiriciCiz(k) {
  const g = $(k.id), oto = k.oto && A[k.id] === null;
  const od = otoDeger(k);
  const v = oto ? (od ?? (k.id === "yogunluk" ? 0.8 : k.id.endsWith("_ms") ? 400 : 1)) : A[k.id];
  g.value = v;
  g.style.setProperty("--yuzde", ((v - k.min) / (k.max - k.min) * 100) + "%");
  g.classList.toggle("oto-mod", oto);
  if (k.oto) $(k.id + "Oto").classList.toggle("acik", oto);
  $(k.id + "Deger").textContent = oto ? (od !== null ? k.goster(Number(od)) : "metne göre") : k.goster(Number(v));
}
function ayarCiz() {
  if (!BILGI) return;
  const s = sesBul(A.ses);
  $("sesSecici").innerHTML = `${avatar(s)}<div><div class="ad">${esc(s.gorunen)}</div><div class="ac">${esc(s.aciklama)}</div></div><span class="ok">${IKON.ok}</span>`;
  if (A.ton !== "otomatik" && !s.tonlar.includes(A.ton)) A.ton = "otomatik";
  $("tonlar").innerHTML = [`<button class="secenek${A.ton === "otomatik" ? " secili" : ""}" data-ton="otomatik">Otomatik</button>`,
    ...s.tonlar.map((t) => `<button class="secenek${A.ton === t ? " secili" : ""}" data-ton="${t}"><span class="nokta" style="background:var(--t-${t})"></span>${esc(buyuk(BILGI.tonlar[t]))}</button>`)].join("");
  $("tonlar").querySelectorAll("[data-ton]").forEach((b) => b.onclick = () => { A.ton = b.dataset.ton; ayarCiz(); });
  $("tonAciklama").textContent = A.ton === "otomatik"
    ? "Her cümlenin duygusu metinden seçilir: soru merakla, uyarı uyarıyla, sevinç neşeyle okunur. Metindeki işaretler bunu ezer."
    : `Metnin tamamı "${BILGI.tonlar[A.ton]}" okunur. Metindeki işaretler ([üzgün] gibi) yine geçerlidir.`;
  $("duyguKisa").textContent = A.ton === "otomatik" ? "cümleye göre" : "bütün metin";
  const otoAd = "Otomatik" + (TAHMIN ? ` · ${buyuk(TAHMIN.ad)}` : "");
  // araç çubuğunda kısa: "Masal (oto)" — tek satıra sığsın
  $("uslupEtiket").textContent = A.uslup !== "otomatik" ? buyuk(BILGI.usluplar[A.uslup].ad)
    : TAHMIN ? `${buyuk(TAHMIN.ad)} (oto)` : "Otomatik";
  $("uslupDugme").title = A.uslup === "otomatik" && TAHMIN ? `Metinden seçildi (${TAHMIN.gerekce}). Değiştirmek için tıklayın.`
    : "Metnin hangi biçimde okunacağı: hız, ezgi, duraklamalar";
  $("uslupListe").innerHTML = [["otomatik", otoAd, "Metnin türüne göre kendisi seçer"],
    ...Object.entries(BILGI.usluplar).map(([k, u]) => [k, buyuk(u.ad), `${USLUP_ACIKLAMA[k] || ""} · hız ×${u.hiz}, ezgi ×${u.tonlama}`])]
    .map(([k, ad, alt]) => `<button class="${A.uslup === k ? "secili" : ""}" data-uslup="${k}"><span>${esc(ad)}</span><span class="alt">${esc(alt)}</span></button>`).join("");
  $("uslupListe").querySelectorAll("[data-uslup]").forEach((b) => b.onclick = () => { A.uslup = b.dataset.uslup; menuleriKapat(); ayarCiz(); });
  $("usluplar").innerHTML = [`<button class="secenek${A.uslup === "otomatik" ? " secili" : ""}" data-uslup="otomatik">${esc(otoAd)}</button>`,
    ...Object.entries(BILGI.usluplar).map(([k, u]) => `<button class="secenek${A.uslup === k ? " secili" : ""}" data-uslup="${k}">${esc(buyuk(u.ad))}</button>`)].join("");
  $("usluplar").querySelectorAll("[data-uslup]").forEach((b) => b.onclick = () => { A.uslup = b.dataset.uslup; ayarCiz(); });
  KAYDIRICI.forEach(kaydiriciCiz);
  yerel.koy("sa.ayar", JSON.stringify(A));
}
$("sifirla").onclick = () => { A = { ...VARSAYILAN, ses: A.ses }; ayarCiz(); };
const istekAyari = () => { const a = { ...A }; for (const k of ["duraklama_ms", "virgul_ms"]) if (a[k] !== null) a[k] = Math.round(a[k]); return a; };

// ── pencereler ve açılır menüler ────────────────────────────────
function perdeAc(id) { $(id).classList.add("acik"); }
function perdeKapat() { document.querySelectorAll(".perde").forEach((p) => p.classList.remove("acik")); if (ornekSes) ornekSes.pause(); }
document.querySelectorAll(".perde").forEach((p) => p.addEventListener("click", (e) => { if (e.target === p || e.target.closest("[data-kapat]")) perdeKapat(); }));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { perdeKapat(); menuleriKapat(); } });
$("sesSecici").onclick = () => { sesIzgarasi($("sesModalIzgara"), true); perdeAc("sesPerde"); };
function menuleriKapat() { document.querySelectorAll(".acilir.acik").forEach((m) => m.classList.remove("acik")); }
document.querySelectorAll("[data-ac]").forEach((b) => b.onclick = (e) => {
  e.stopPropagation(); const m = $(b.dataset.ac); const acik = m.classList.contains("acik"); menuleriKapat(); if (!acik) m.classList.add("acik"); });
document.addEventListener("click", menuleriKapat);

// ── düzenleyici: işaret ekleme, sayaç, okunuş ───────────────────
function imleceEkle(yazi) {
  const t = $("metin"), bas = t.selectionStart ?? t.value.length, son = t.selectionEnd ?? bas;
  const once = t.value.slice(0, bas), sonra = t.value.slice(son);
  const ara = (once && !/\s$/.test(once) ? " " : "") + yazi + (sonra && !/^\s/.test(sonra) ? " " : "");
  t.value = once + ara + sonra;
  const yeni = once.length + ara.length;
  t.focus(); t.setSelectionRange(yeni, yeni);
  sayacCiz(); menuleriKapat();
}
document.querySelectorAll("[data-ekle]").forEach((b) => b.onclick = () => imleceEkle(b.dataset.ekle));
$("durakEkle").onclick = () => imleceEkle("[duraklama 1]");

const sure = (s) => s >= 3600 ? `${Math.floor(s / 3600)}:${String(Math.floor(s % 3600 / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`
  : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
function sayacCiz() {
  const v = $("metin").value, temiz = v.replace(/\[[^\]\n]{1,40}\]/g, " ");
  const harf = (temiz.match(/[A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû0-9]/g) || []).length;
  const paragraf = v.trim() ? v.trim().split(/\n\s*\n+/).length : 0;
  $("sayac").textContent = `${v.length.toLocaleString("tr-TR")} / ${(BILGI?.en_cok || 50000).toLocaleString("tr-TR")} karakter`
    + (paragraf > 1 ? ` · ${paragraf} paragraf` : "") + (harf ? ` · yaklaşık ${sure(harf / 13.5)}` : "");
  yerel.koy("sa.metin", v);
  if ($("okunusKutu").classList.contains("acik")) okunusZamanla();
  uslupTahmin();
}
$("metin").addEventListener("input", sayacCiz);
// Otomatik üslup önizlemesi: yazmaya ara verince sunucu metnin türünü söyler (seslendirmede aynı karar verilir)
const USLUP_ACIKLAMA = { haber: "Tarafsız, akıcı", bilgi: "Açık, ölçülü", talimat: "Adım adım, net", deneme: "Düşünceli, ağırbaşlı",
  anlati: "Hikâye anlatır gibi", masal: "Yavaş, canlı, sıcak", sohbet: "Doğal, rahat" };
let TAHMIN = null, tahminSaat = null, tahminNo = 0;
function uslupTahmin() {
  clearTimeout(tahminSaat);
  tahminSaat = setTimeout(async () => {
    const metin = $("metin").value.trim(), no = ++tahminNo;
    if (!metin) { TAHMIN = null; ayarCiz(); return; }
    try {
      const y = await fetch("/api/uslup", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ metin: metin.slice(0, 4000) }) });
      if (y.ok && no === tahminNo) { TAHMIN = await y.json(); ayarCiz(); }
    } catch { /* sunucu yoksa önizleme yok */ }
  }, 600);
}

$("metin").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) seslendir(); });
$("temizle").onclick = () => { $("metin").value = ""; sayacCiz(); $("metin").focus(); };

let okunusZaman = null;
function okunusZamanla() { clearTimeout(okunusZaman); okunusZaman = setTimeout(okunusGoster, 450); }
async function okunusGoster() {
  const metin = $("metin").value.trim();
  const kutu = $("okunusKutu");
  if (!metin) { kutu.innerHTML = `<div class="aciklama">Metin boş.</div>`; return; }
  try {
    const r = await fetch("/api/okunus", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ metin: metin.slice(0, 20000) }) });
    const j = await r.json();
    kutu.innerHTML = `<div class="aciklama" style="margin:0 0 6px">Kalın ve altı çizili: vurgulu hece. Sayılar, tarihler ve kısaltmalar okunacakları gibi yazılır.</div>` +
      j.cumleler.map((c) => `<div class="ok-satir"><div class="ozgun">${esc(c.cumle)}</div><div class="okunan">${c.sozcukler.map((w) =>
        w.v ? esc(w.s.slice(0, w.v[0])) + "<b>" + esc(w.s.slice(w.v[0], w.v[1])) + "</b>" + esc(w.s.slice(w.v[1])) : esc(w.s)).join(" ")}</div>
        ${c.telaffuz.length ? `<div class="tel">telaffuz sözlüğünden: ${c.telaffuz.map(esc).join(", ")}</div>` : ""}</div>`).join("") +
      (j.cumleler.length >= 60 ? `<div class="aciklama">İlk 60 cümle gösteriliyor.</div>` : "");
  } catch (_) { kutu.innerHTML = `<div class="aciklama">Okunuş alınamadı.</div>`; }
}
$("okunusDugme").onclick = () => {
  const acik = $("okunusKutu").classList.toggle("acik");
  $("okunusDugme").classList.toggle("secili", acik);
  if (acik) okunusGoster();
};

// ── seslendirme ve oynatıcı ─────────────────────────────────────
let ctx = null, kaynaklar = [], iptal = null, gecmis = [], aktif = null, baslangic = 0, toplamSure = 0, saatId = null;
function durum(yazi, hata = false) { $("durum").textContent = yazi; $("durum").className = "durum" + (hata ? " hata" : ""); }
function sustur() {
  for (const k of kaynaklar) { try { k.stop(); } catch (_) {} }
  kaynaklar = [];
  document.querySelectorAll(".cumleler li.caliyor").forEach((x) => x.classList.remove("caliyor"));
  clearInterval(saatId); saatId = null;
  $("oynat").innerHTML = IKON.oynat;
}
function zamanla(kayit, basIndeks = 0, tek = false) {
  sustur();
  if (ctx.state === "suspended") ctx.resume();
  let t = ctx.currentTime + 0.08;
  baslangic = t; toplamSure = 0;
  kayit.parcalar.forEach((p, i) => {
    if (i < basIndeks || (tek && i !== basIndeks)) return;
    const li = p.li;
    const k = ctx.createBufferSource(); k.buffer = p.ses; k.connect(ctx.destination); k.start(t);
    const bas = t;
    setTimeout(() => li?.classList.add("caliyor"), Math.max(0, (bas - ctx.currentTime) * 1000));
    k.onended = () => li?.classList.remove("caliyor");
    kaynaklar.push(k);
    t += p.ses.duration + (tek ? 0 : p.sessizlik_ms / 1000);
  });
  toplamSure = t - baslangic;
  saatBaslat();
}
function saatBaslat() {
  $("oynat").innerHTML = IKON.durdur;
  clearInterval(saatId);
  saatId = setInterval(() => {
    const g = Math.max(0, ctx.currentTime - baslangic);
    $("dolgu").style.width = Math.min(100, g / Math.max(toplamSure, 0.01) * 100) + "%";
    $("zaman").textContent = sure(Math.min(g, toplamSure));
    $("toplam").textContent = sure(toplamSure);
    if (g >= toplamSure && !iptal) { clearInterval(saatId); saatId = null; $("oynat").innerHTML = IKON.oynat; }
  }, 150);
}
$("oynat").onclick = () => {
  if (!aktif || !ctx) return;
  if (saatId && ctx.state === "running") { ctx.suspend(); $("oynat").innerHTML = IKON.oynat; return; }
  if (saatId && ctx.state === "suspended") { ctx.resume(); $("oynat").innerHTML = IKON.durdur; return; }
  zamanla(aktif);
};
function uslupCiz(kayit, ek = "") {
  if (!kayit.okuma) { $("uslupSatir").innerHTML = ""; return; }
  $("uslupSatir").innerHTML = `<span class="rozet">${esc(buyuk(BILGI.usluplar[kayit.uslup]?.ad || kayit.uslup))}</span>
    <span>${esc(kayit.uslupNeden)} · hız ×${kayit.okuma.hiz} · tonlama ×${kayit.okuma.tonlama}${ek}</span>`;
  $("uyariSatir").hidden = !kayit.uyarilar?.length;
  $("uyariSatir").textContent = kayit.uyarilar?.length ? "⚠ " + kayit.uyarilar.join(" · ") + " (sesin içinde okunmadı)" : "";
}
function cumleEkle(p, i, kayit) {
  if (i === 0 || p.paragraf !== kayit.parcalar[i - 1].paragraf) {
    if (i > 0 || kayit.parcalar.some((x) => x.paragraf > 0) || p.paragraf > 0) {
      const ph = document.createElement("li"); ph.className = "paragraf"; ph.textContent = `Paragraf ${p.paragraf + 1}`;
      $("cumleler").appendChild(ph);
    }
  }
  const li = document.createElement("li");
  li.innerHTML = `${tonRozet(p.ton, p.yogunluk)}<span></span><span class="neden"></span>`;
  li.children[1].textContent = p.cumle;
  li.children[2].textContent = p.neden;
  li.title = "Bu cümleyi çal";
  li.onclick = () => { if (aktif) zamanla(aktif, i, true); };
  p.li = li;
  $("cumleler").appendChild(li);
}
function kayitGoster(kayit) {
  aktif = kayit;
  $("oynatici").hidden = false;
  uslupCiz(kayit);
  $("cumleler").innerHTML = "";
  kayit.parcalar.forEach((p, i) => cumleEkle(p, i, kayit));
}

async function seslendir() {
  const metin = $("metin").value.trim();
  if (!metin) { durum("Önce bir metin yazın.", true); $("metin").focus(); return; }
  if (iptal) iptal.abort();
  ctx = ctx || new AudioContext();
  sustur();
  if (ctx.state === "suspended") await ctx.resume();
  const kayit = { metin, ses: A.ses, ayar: { ...A }, parcalar: [], zaman: new Date(), okuma: null, uslup: "", uslupNeden: "", uyarilar: [] };
  aktif = kayit;
  $("oynatici").hidden = false; $("cumleler").innerHTML = ""; $("uslupSatir").innerHTML = ""; $("uyariSatir").hidden = true; $("dolgu").style.width = "0";
  $("uret").disabled = true; $("durdur").hidden = false; durum("Hazırlanıyor…");
  iptal = new AbortController();
  const t0 = performance.now();
  let t = ctx.currentTime + 0.1; baslangic = t; toplamSure = 0;
  try {
    const r = await fetch("/api/seslendir", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ metin, ayar: istekAyari() }), signal: iptal.signal });
    if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(typeof e.detail === "string" ? e.detail : "İstek reddedildi; metni ve ayarları denetleyin."); }
    const okuyucu = r.body.getReader(), coz = new TextDecoder();
    let tampon = "";
    saatBaslat();
    for (;;) {
      const { value, done } = await okuyucu.read();
      if (done) break;
      tampon += coz.decode(value, { stream: true });
      let i;
      while ((i = tampon.indexOf("\n")) >= 0) {
        const satir = tampon.slice(0, i); tampon = tampon.slice(i + 1);
        if (!satir.trim()) continue;
        const p = JSON.parse(satir);
        if (p.tur === "hata") throw new Error(p.ileti);
        if (p.tur !== "parca") continue;
        if (!kayit.okuma) {
          kayit.okuma = p.bilgi.okuma; kayit.uslup = p.bilgi.uslup; kayit.uslupNeden = p.bilgi.uslup_gerekce; kayit.uyarilar = p.bilgi.uyarilar || [];
          uslupCiz(kayit, ` · ilk ses ${((performance.now() - t0) / 1000).toFixed(1)} sn`);
        }
        const ses = await ctx.decodeAudioData(Uint8Array.from(atob(p.wav), (c) => c.charCodeAt(0)).buffer);
        const parca = { cumle: p.cumle, ton: p.ton, yogunluk: p.yogunluk, sessizlik_ms: p.sessizlik_ms, neden: p.bilgi.ton_gerekce, paragraf: p.bilgi.paragraf || 0, ses };
        const n = kayit.parcalar.push(parca) - 1;
        if (aktif === kayit) {
          cumleEkle(parca, n, kayit);
          const k = ctx.createBufferSource(); k.buffer = ses; k.connect(ctx.destination);
          t = Math.max(t, ctx.currentTime + 0.05); k.start(t);
          const bas = t, li = parca.li;
          setTimeout(() => li?.classList.add("caliyor"), Math.max(0, (bas - ctx.currentTime) * 1000));
          k.onended = () => li?.classList.remove("caliyor");
          kaynaklar.push(k);
          t += ses.duration + p.sessizlik_ms / 1000; toplamSure = t - baslangic;
        }
        durum(`${n + 1} cümle hazır`);
      }
    }
    durum(`${kayit.parcalar.length} cümle · ${((performance.now() - t0) / 1000).toFixed(1)} sn`);
    gecmis.unshift(kayit); gecmis = gecmis.slice(0, 8); gecmisCiz();
  } catch (h) {
    durum(h.name === "AbortError" ? "Durduruldu" : (h.message || String(h)), h.name !== "AbortError");
    if (h.name === "AbortError" && kayit.parcalar.length) { gecmis.unshift(kayit); gecmis = gecmis.slice(0, 8); gecmisCiz(); }
  } finally {
    $("uret").disabled = false; $("durdur").hidden = true; iptal = null;
  }
}
$("uret").onclick = seslendir;
$("durdur").onclick = () => { if (iptal) iptal.abort(); sustur(); };

function gecmisCiz() {
  $("gecmisKart").hidden = !gecmis.length;
  $("gecmis").innerHTML = gecmis.map((k, i) => {
    const s = sesBul(k.ses);
    const toplam = k.parcalar.reduce((a, p) => a + p.ses.duration + p.sessizlik_ms / 1000, 0);
    return `<div class="gecmis-oge">${avatar(s, 32)}<div class="metin"><div>${esc(k.metin.replace(/\s+/g, " "))}</div>
      <div>${esc(s.gorunen)} · ${k.ayar.ton === "otomatik" ? "duygu otomatik" : esc(BILGI.tonlar[k.ayar.ton])} · ${k.parcalar.length} cümle · ${sure(toplam)} · ${k.zaman.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" })}</div></div>
      <button class="ikon-dugme" data-ac-kayit="${i}" title="Aç ve çal">${IKON.dinle}</button>
      <button class="dugme kucuk" data-indir="${i}" title="WAV indir">WAV</button>
      <button class="dugme kucuk" data-mp3="${i}" title="MP3 indir">MP3</button></div>`;
  }).join("");
  $("gecmis").querySelectorAll("[data-ac-kayit]").forEach((b) => b.onclick = () => { const k = gecmis[b.dataset.acKayit]; kayitGoster(k); zamanla(k); });
  $("gecmis").querySelectorAll("[data-indir]").forEach((b) => b.onclick = () => indir(gecmis[b.dataset.indir], "wav"));
  $("gecmis").querySelectorAll("[data-mp3]").forEach((b) => b.onclick = () => indir(gecmis[b.dataset.mp3], "mp3"));
}

function wavBlob(kayit) {
  const oran = kayit.parcalar[0].ses.sampleRate;
  let n = 0;
  kayit.parcalar.forEach((p, i) => { n += p.ses.length + (i < kayit.parcalar.length - 1 ? Math.round(p.sessizlik_ms / 1000 * oran) : 0); });
  const veri = new Int16Array(n);
  let i = 0;
  for (const p of kayit.parcalar) {
    const x = p.ses.getChannelData(0);
    for (let j = 0; j < x.length; j++) veri[i++] = Math.max(-1, Math.min(1, x[j])) * 32767;
    i += Math.round(p.sessizlik_ms / 1000 * oran);
  }
  const b = new ArrayBuffer(44 + veri.byteLength), v = new DataView(b);
  const yaz = (o, s) => [...s].forEach((c, k) => v.setUint8(o + k, c.charCodeAt(0)));
  yaz(0, "RIFF"); v.setUint32(4, 36 + veri.byteLength, true); yaz(8, "WAVE"); yaz(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, oran, true);
  v.setUint32(28, oran * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); yaz(36, "data");
  v.setUint32(40, veri.byteLength, true); new Int16Array(b, 44).set(veri.subarray(0, (b.byteLength - 44) / 2));
  return new Blob([b], { type: "audio/wav" });
}
async function indir(kayit = aktif, bicim = "wav") {
  if (!kayit || !kayit.parcalar.length) return;
  let blob = wavBlob(kayit);
  if (bicim === "mp3") {
    durum("MP3'e dönüştürülüyor…");
    const r = await fetch("/api/donustur?bicim=mp3", { method: "POST", body: blob });
    if (!r.ok) { durum("MP3'e dönüştürülemedi.", true); return; }
    blob = await r.blob(); durum("MP3 hazır");
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `seslen-${kayit.ses}-${kayit.zaman.toISOString().slice(0, 19).replace(/[:T]/g, "-")}.${bicim}`;
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 8000);
}
$("indirWav").onclick = () => indir(aktif, "wav");
$("indirMp3").onclick = () => indir(aktif, "mp3");

// ── telaffuz sözlüğü ────────────────────────────────────────────
let SOZLUK = {};
function telaffuzCiz() {
  const anahtar = Object.keys(SOZLUK);
  $("telSayi").textContent = anahtar.length || "";
  $("telListe").innerHTML = anahtar.length ? anahtar.map((y) => `<div class="tel-satir"><span class="y">${esc(y)}</span><span class="ok">→</span>
      <span class="o">${esc(SOZLUK[y])}</span><span class="eylem"><button class="dugme kucuk" data-tdinle="${esc(y)}">${IKON.dinle} Dinle</button>
      <button class="ikon-dugme" data-tsil="${esc(y)}" title="Sil">${IKON.sil}</button></span></div>`).join("")
    : `<div class="bos">Sözlük boş. Yukarıdan ilk kaydı ekleyin; örneğin <b>GIS</b> → <b>ci ay es</b>.</div>`;
  $("telListe").querySelectorAll("[data-tsil]").forEach((b) => b.onclick = async () => {
    const r = await fetch(`/api/telaffuz/${encodeURIComponent(b.dataset.tsil)}`, { method: "DELETE" });
    if (r.ok) { SOZLUK = (await r.json()).sozluk; telaffuzCiz(); }
  });
  $("telListe").querySelectorAll("[data-tdinle]").forEach((b) => b.onclick = () => kisaDinle(`${b.dataset.tdinle}.`, b));
}
async function kisaDinle(metin, dugme) {
  const eski = dugme.innerHTML; dugme.disabled = true; dugme.innerHTML = "…";
  try {
    const r = await fetch("/v1/audio/speech", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: metin, voice: A.ses, ton: "notr" }) });
    if (!r.ok) throw new Error();
    const ses = new Audio(URL.createObjectURL(await r.blob()));
    ses.onended = ses.onerror = () => { dugme.innerHTML = eski; dugme.disabled = false; };
    await ses.play();
  } catch (_) { dugme.innerHTML = eski; dugme.disabled = false; }
}
$("telEkle").onclick = async () => {
  const yazilis = $("telYazilis").value.trim(), okunus = $("telOkunus").value.trim();
  if (!yazilis || !okunus) { $("telDurum").textContent = "İki alanı da doldurun."; return; }
  const r = await fetch("/api/telaffuz", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ yazilis, okunus }) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) { $("telDurum").textContent = typeof j.detail === "string" ? j.detail : "Eklenemedi."; return; }
  SOZLUK = j.sozluk; telaffuzCiz();
  $("telDurum").textContent = `"${yazilis}" artık "${okunus}" diye okunacak.`;
  $("telYazilis").value = ""; $("telOkunus").value = ""; $("telYazilis").focus();
};
$("telDinle").onclick = () => { const o = $("telOkunus").value.trim() || $("telYazilis").value.trim(); if (o) kisaDinle(o + ".", $("telDinle")); };
$("telOkunus").addEventListener("keydown", (e) => { if (e.key === "Enter") $("telEkle").click(); });

// ── ana sayfa bölümleri ─────────────────────────────────────────
function anaSayfaCiz() {
  $("isaretler").innerHTML = BILGI.isaretler.map((x) => `<button class="kart isaret-kart" data-isaret="${esc(x.ornek)}"><code>${esc(x.ornek)}</code><div>${esc(x.ne)}</div></button>`).join("");
  $("isaretler").querySelectorAll("[data-isaret]").forEach((b) => b.onclick = () => {
    const i = b.dataset.isaret;
    if (i === "boş satır") imleceEkle("\n\n"); else imleceEkle(i);
    $("metin").scrollIntoView({ behavior: "smooth", block: "center" });
  });
  $("bilesenler").innerHTML = BILGI.bilesenler.map((b, i) => `<div class="kart bilesen"><span class="sira">${i + 1}</span>
    <div class="ad">${esc(b.ad)}</div><div class="is">${esc(b.is)}</div><span class="olcu">${esc(b.olcu)}</span></div>`).join("");
  const sec = ["damla", "serdar", "fatma", "pitircik"].map(sesBul).filter(Boolean);
  $("oneCikan").innerHTML = sec.map((s) => `<div class="kart mini-ses">${avatar(s, 42)}<div class="metin"><div class="ad">${esc(s.gorunen)}</div>
    <div class="ac">${esc(s.aciklama)}</div></div><button class="ikon-dugme" data-mdinle="${s.ad}" title="Dinle">${IKON.dinle}</button>
    <button class="dugme kucuk" data-msec="${s.ad}">Seç</button></div>`).join("");
  $("oneCikan").querySelectorAll("[data-mdinle]").forEach((b) => b.onclick = () => ornekCal(b.dataset.mdinle, b));
  $("oneCikan").querySelectorAll("[data-msec]").forEach((b) => b.onclick = () => { sesSec(b.dataset.msec); window.scrollTo({ top: 0, behavior: "smooth" }); });
  $("duyguListe").innerHTML = [...Object.entries(ISARET_AD).map(([k, ad]) => `<button data-ekle="[${ad}]"><span class="nokta" style="background:var(--t-${k})"></span>${esc(buyuk(ad))}<small>[${esc(ad)}]</small></button>`),
    `<button data-ekle="[otomatik]">Otomatiğe dön<small>[otomatik]</small></button>`].join("");
  $("duyguListe").querySelectorAll("[data-ekle]").forEach((b) => b.onclick = () => imleceEkle(b.dataset.ekle));
}
for (const [ad, metin] of Object.entries(ORNEK)) {
  const b = document.createElement("button"); b.className = "cip"; b.textContent = ad;
  b.onclick = () => { $("metin").value = metin; sayacCiz(); };
  $("ornekler").appendChild(b);
}

// ── API sayfası ─────────────────────────────────────────────────
const KOD = {
  python: (u) => `import requests

yanit = requests.post("${u}/v1/audio/speech", json={
    "input": "[neşeli] Merhaba! [duraklama 0.5] Bugün hava çok güzel, parka gidelim mi?",
    "voice": "zeynep",
    "uslup": "otomatik",          # ya da: haber, bilgi, talimat, deneme, anlati, masal, sohbet
    "response_format": "mp3",     # ya da: wav
})
yanit.raise_for_status()
open("ses.mp3", "wb").write(yanit.content)
print(yanit.headers["X-Tonlar"])  # cümle başına seçilen duygular`,
  js: (u) => `const yanit = await fetch("${u}/v1/audio/speech", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    input: "Merhaba! Bugün hava çok güzel, parka gidelim mi?",
    voice: "zeynep",
    ton: "neseli",          // bütün metin tek duyguda
    hiz: 1.05,
  }),
});
if (!yanit.ok) throw new Error(await yanit.text());
new Audio(URL.createObjectURL(await yanit.blob())).play();`,
  curl: (u) => `# istek.json:
# {"input": "Merhaba! Bugün hava çok güzel.", "voice": "kuzey", "uslup": "haber"}

curl -X POST ${u}/v1/audio/speech \\
  -H "Content-Type: application/json; charset=utf-8" \\
  --data-binary @istek.json \\
  -o ses.wav`,
};
let kodDil = "python";
function kodCiz() {
  $("kod").textContent = KOD[kodDil](location.origin);
  document.querySelectorAll("#kodSekme [data-dil]").forEach((b) => b.classList.toggle("etkin", b.dataset.dil === kodDil));
}
document.querySelectorAll("#kodSekme [data-dil]").forEach((b) => b.onclick = () => { kodDil = b.dataset.dil; kodCiz(); });
$("kopyala").onclick = async () => {
  try { await navigator.clipboard.writeText($("kod").textContent); $("kopyala").textContent = "Kopyalandı"; }
  catch (_) { $("kopyala").textContent = "Kopyalanamadı"; }
  setTimeout(() => $("kopyala").textContent = "Kopyala", 1500);
};

// ── açılış ──────────────────────────────────────────────────────
async function ac() {
  kaydiriciKur();
  $("oynat").innerHTML = IKON.oynat;
  BILGI = await (await fetch("/api/bilgi")).json();
  $("metin").maxLength = BILGI.en_cok;
  $("modelBilgi").innerHTML = `<b>${esc(BILGI.model.replace(/\.pth$/, ""))}</b><br>${BILGI.sesler.length} ses · ${Object.keys(BILGI.tonlar).length} duygu${BILGI.vurgu ? " · vurgulu okunuş" : ""}<br>işlemcide, internetsiz`;
  $("sesSayi").textContent = BILGI.sesler.length; $("istSes").textContent = BILGI.sesler.length;
  try { const k = JSON.parse(yerel.al("sa.ayar") || "null"); if (k && sesBul(k.ses)) for (const a in VARSAYILAN) if (a in k) A[a] = k[a]; } catch (_) {}
  if (!sesBul(A.ses)) A.ses = BILGI.sesler[0].ad;
  $("metin").value = yerel.al("sa.metin") || ORNEK["Masal"];
  sayacCiz();
  ayarCiz();
  anaSayfaCiz();
  SOZLUK = (await (await fetch("/api/telaffuz")).json()).sozluk; telaffuzCiz();
  $("temelAdres").textContent = location.origin;
  $("apiSinir").textContent = BILGI.en_cok.toLocaleString("tr-TR");
  $("apiSesler").innerHTML = BILGI.sesler.map((s) => `<code>${s.ad}</code>`).join(" ");
  $("apiTonlar").innerHTML = Object.keys(BILGI.tonlar).map((t) => `<code>${t}</code>`).join(" ");
  $("apiUsluplar").innerHTML = Object.keys(BILGI.usluplar).map((t) => `<code>${t}</code>`).join(" ");
  kodCiz();
  git();
}
ac().catch((h) => durum("Sunucuya ulaşılamadı: " + h.message, true));
