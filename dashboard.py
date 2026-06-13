# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  DASHBOARD SWING TRADING — Opportunità & Portafoglio
  Compagna di fase6_standalone (legge segnali.db).
  Tutti i dati di mercato provengono da Yahoo Finance (fonte ufficiale,
  la stessa usata dall'algoritmo): prezzo, RSI, Stocastico, P/E, P/B,
  supporti/resistenze e news con sentiment. News filtrate alle ultime 24h.
═══════════════════════════════════════════════════════════════════════

AVVIO:
    pip install streamlit yfinance pandas vaderSentiment
    streamlit run dashboard.py
"""

import os
import re
import json
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

# ── Percorsi (relativi alla cartella di questo file) ───────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DB_PATH        = os.path.join(BASE_DIR, "segnali.db")
PORTFOLIO_PATH = os.path.join(BASE_DIR, "portafoglio.json")
GUADAGNI_PATH  = os.path.join(BASE_DIR, "guadagni.json")

st.set_page_config(page_title="Swing Trading Dashboard", page_icon="📈", layout="wide")


# ── Modalità SOLA VISUALIZZAZIONE ──────────────────────────────────────
# Attivala sul deploy condiviso (Streamlit Cloud) impostando il secret
# VIEW_ONLY = "true": il socio vede tutto ma non può modificare il portafoglio.
# In locale (senza il flag) hai il controllo completo.
def _flag_view_only() -> bool:
    try:
        if "VIEW_ONLY" in st.secrets:
            return str(st.secrets["VIEW_ONLY"]).lower() in ("1", "true", "yes", "si", "sì")
    except Exception:
        pass
    return str(os.environ.get("VIEW_ONLY", "0")).lower() in ("1", "true", "yes", "si", "sì")


SOLO_VISUALIZZAZIONE = _flag_view_only()


# ═══════════════════════════════════════════════════════════════════════
#  DATI DI MERCATO (Yahoo Finance) — prezzo, indicatori, fondamentali
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def scarica_storico(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Storico giornaliero da Yahoo Finance (cache 10 min)."""
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def _rsi(close: pd.Series, period: int = 14):
    """RSI di Wilder sull'ultima barra."""
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - 100 / (1 + rs)
    v = float(rsi.iloc[-1])
    return round(v, 1) if v == v else None


def _bollinger(close: pd.Series, period: int = 20, mult: float = 2.0):
    """Bande di Bollinger (SMA20 ± 2σ). Ritorna (inf, mid, sup, %B, posizione)."""
    if len(close) < period:
        return None, None, None, None, "—"
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    mid = float(sma.iloc[-1])
    sup = mid + mult * float(std.iloc[-1])
    inf = mid - mult * float(std.iloc[-1])
    prezzo = float(close.iloc[-1])
    pct_b = (prezzo - inf) / (sup - inf) if sup > inf else None      # %B: 0=banda inf, 1=banda sup
    if prezzo >= sup:
        pos = "🔴 sopra"      # tocca/supera banda alta → ipercomprato/breakout
    elif prezzo <= inf:
        pos = "🟢 sotto"      # tocca/supera banda bassa → ipervenduto
    else:
        pos = "⚪ dentro"
    return (round(inf, 3), round(mid, 3), round(sup, 3),
            round(pct_b * 100, 1) if pct_b is not None else None, pos)


def _stocastico(df: pd.DataFrame, k: int = 14, d: int = 3):
    """Stocastico %K (e %D = media mobile di %K). Ritorna (%K, %D) sull'ultima barra."""
    if len(df) < k + d:
        return None, None
    low_k = df["Low"].rolling(k).min()
    high_k = df["High"].rolling(k).max()
    pk = 100 * (df["Close"] - low_k) / (high_k - low_k).replace(0, 1e-9)
    pd_line = pk.rolling(d).mean()
    a = float(pk.iloc[-1]); b = float(pd_line.iloc[-1])
    return (round(a, 1) if a == a else None,
            round(b, 1) if b == b else None)


def _clusterizza(livelli, tol=0.012):
    """Unisce livelli entro `tol` (1.2%) tenendo la media del cluster."""
    if not livelli:
        return []
    livelli = sorted(livelli)
    cluster, out = [livelli[0]], []
    for v in livelli[1:]:
        if abs(v - cluster[-1]) / cluster[-1] <= tol:
            cluster.append(v)
        else:
            out.append(sum(cluster) / len(cluster))
            cluster = [v]
    out.append(sum(cluster) / len(cluster))
    return out


def _supporti_resistenze(df: pd.DataFrame, prezzo: float, n: int = 2) -> dict:
    """Primi `n` supporti (sotto) e resistenze (sopra): swing high/low + pivot."""
    if df.empty:
        return {"supporti": [], "resistenze": []}
    highs, lows = df["High"].values, df["Low"].values
    w = 5
    shi, slo = [], []
    for i in range(w, len(df) - w):
        if highs[i] == highs[i - w:i + w + 1].max():
            shi.append(float(highs[i]))
        if lows[i] == lows[i - w:i + w + 1].min():
            slo.append(float(lows[i]))
    h, l, c = float(df["High"].iloc[-1]), float(df["Low"].iloc[-1]), float(df["Close"].iloc[-1])
    piv = (h + l + c) / 3
    shi += [2 * piv - l, piv + (h - l)]
    slo += [2 * piv - h, piv - (h - l)]
    res = sorted(_clusterizza([v for v in shi if v > prezzo * 1.002]))[:n]
    sup = sorted(_clusterizza([v for v in slo if v < prezzo * 0.998]), reverse=True)[:n]
    return {"supporti": [round(x, 3) for x in sup],
            "resistenze": [round(x, 3) for x in res]}


@st.cache_data(ttl=300, show_spinner=False)
def metriche_mercato(ticker: str) -> dict:
    """
    Prezzo attuale + RSI + Stocastico + primo supporto/resistenza, da un unico
    download dello storico Yahoo Finance (cache 5 min).
    """
    out = {"prezzo": None, "rsi": None, "stoch_k": None, "stoch_d": None,
           "sup1": None, "res1": None, "supporti": [], "resistenze": [],
           "bb_inf": None, "bb_mid": None, "bb_sup": None, "bb_pct": None, "bb_pos": "—"}
    df = scarica_storico(ticker, period="6mo")
    if df.empty:
        return out
    out["prezzo"] = round(float(df["Close"].iloc[-1]), 4)
    out["rsi"] = _rsi(df["Close"])
    out["stoch_k"], out["stoch_d"] = _stocastico(df)
    out["bb_inf"], out["bb_mid"], out["bb_sup"], out["bb_pct"], out["bb_pos"] = _bollinger(df["Close"])
    sr = _supporti_resistenze(df, out["prezzo"])
    out["supporti"], out["resistenze"] = sr["supporti"], sr["resistenze"]
    out["sup1"] = sr["supporti"][0] if sr["supporti"] else None
    out["res1"] = sr["resistenze"][0] if sr["resistenze"] else None
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def _ha_dati(simbolo: str) -> bool:
    """True se il simbolo restituisce dati di prezzo su Yahoo Finance."""
    return not scarica_storico(simbolo, period="5d").empty


@st.cache_data(ttl=86400, show_spinner=False)
def risolvi_simbolo(isin: str = "", ticker_o_nome: str = "") -> dict:
    """
    Risolve il simbolo ufficiale Yahoo Finance partendo da ISIN (preferito) o
    da ticker/nome. Usa l'endpoint di ricerca ufficiale Yahoo, che accetta ISIN.
    Ritorna {'simbolo': str|None, 'fonte': str, 'nome': str}.
    """
    import requests
    candidati = [c.strip() for c in (isin, ticker_o_nome) if c and c.strip()]

    # 1) Se il ticker inserito è già valido su Yahoo, accettalo direttamente.
    t = (ticker_o_nome or "").strip().upper()
    if t and _ha_dati(t):
        return {"simbolo": t, "fonte": "ticker diretto", "nome": t}

    # 2) Ricerca Yahoo per ISIN o nome (la prima query che produce un match vince).
    for q in candidati:
        try:
            r = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": q, "quotesCount": 6, "newsCount": 0},
                headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
            )
            quotes = (r.json() or {}).get("quotes", [])
            # preferisci strumenti azionari (EQUITY); poi qualsiasi simbolo valido
            ordinati = sorted(quotes, key=lambda x: 0 if x.get("quoteType") == "EQUITY" else 1)
            for qd in ordinati:
                sym = qd.get("symbol")
                if sym and _ha_dati(sym):
                    nome = qd.get("shortname") or qd.get("longname") or sym
                    fonte = "ISIN" if q == isin.strip() else "ricerca nome"
                    return {"simbolo": sym, "fonte": fonte, "nome": nome}
        except Exception:
            continue
    return {"simbolo": None, "fonte": "non risolto", "nome": ticker_o_nome or isin}


@st.cache_data(ttl=1800, show_spinner=False)
def fondamentali(ticker: str) -> dict:
    """Nome, settore, P/E (trailing) e P/B da Yahoo Finance (cache 30 min)."""
    out = {"nome": None, "settore": None, "pe": None, "pb": None}
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        out["nome"] = info.get("longName") or info.get("shortName")
        out["settore"] = info.get("sector")
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        out["pe"] = round(float(pe), 2) if isinstance(pe, (int, float)) else None
        out["pb"] = round(float(pb), 2) if isinstance(pb, (int, float)) else None
    except Exception:
        pass
    return out


# ═══════════════════════════════════════════════════════════════════════
#  NEWS + SENTIMENT (ultime 24h) — Yahoo Finance + VADER finance-aware
# ═══════════════════════════════════════════════════════════════════════

# Lessico finanziario compatto (Loughran-McDonald-lite) per correggere VADER
_LM_POS = {"beat", "beats", "upgrade", "upgraded", "outperform", "buyback", "record",
           "surge", "surges", "strong", "raises", "raised", "tops", "exceeds",
           "approval", "approved", "expansion", "profit", "rebound", "wins", "robust"}
_LM_NEG = {"miss", "misses", "missed", "downgrade", "downgraded", "underperform",
           "cuts", "cut", "warning", "warns", "probe", "lawsuit", "investigation",
           "dilution", "default", "bankruptcy", "recall", "fraud", "delay", "plunge",
           "plunges", "slump", "weak", "loss", "losses", "layoffs", "writedown"}


@st.cache_resource(show_spinner=False)
def _vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def _sentiment_titolo(titolo: str) -> float:
    """Sentiment [-1,+1] VADER corretto col lessico finanziario."""
    an = _vader()
    v = an.polarity_scores(titolo)["compound"] if an else 0.0
    testo = (titolo or "").lower()
    lm = sum(1 for w in _LM_POS if w in testo) - sum(1 for w in _LM_NEG if w in testo)
    if lm < 0 and v > 0.10:        # VADER positivo ma notizia finanziaria negativa
        v = v * 0.3 + lm * 0.15
    elif lm > 0 and v < -0.10:
        v = v * 0.5 + lm * 0.10
    elif lm != 0:
        v = v + 0.05 * lm
    return max(-1.0, min(1.0, v))


def _epoch_da_news(item: dict):
    """Estrae timestamp UTC dall'item news (gestisce schema vecchio e nuovo yfinance)."""
    if "providerPublishTime" in item:
        try:
            return datetime.fromtimestamp(item["providerPublishTime"], tz=timezone.utc)
        except Exception:
            return None
    cont = item.get("content") or {}
    pub = cont.get("pubDate") or cont.get("displayTime")
    if pub:
        try:
            return datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def _campi_news(item: dict):
    """Titolo, publisher, link, timestamp dall'item (schema vecchio/nuovo)."""
    ts = _epoch_da_news(item)
    if "title" in item:
        return (item.get("title", ""), item.get("publisher", ""),
                item.get("link", ""), ts)
    cont = item.get("content") or {}
    prov = (cont.get("provider") or {}).get("displayName", "")
    url = ((cont.get("canonicalUrl") or {}).get("url")
           or (cont.get("clickThroughUrl") or {}).get("url", ""))
    return (cont.get("title", ""), prov, url, ts)


@st.cache_data(ttl=900, show_spinner=False)
def news_sentiment(ticker: str, ore: int = 24) -> dict:
    """News Yahoo Finance delle ultime `ore` ore + sentiment medio finance-aware."""
    try:
        import yfinance as yf
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=ore)
    recenti = []
    for it in items:
        titolo, publisher, link, ts = _campi_news(it)
        if not titolo:
            continue
        if ts is not None and ts < cutoff:
            continue   # più vecchia di `ore` → scartata
        recenti.append({
            "titolo": titolo,
            "publisher": publisher,
            "link": link,
            "quando": ts.astimezone().strftime("%d/%m %H:%M") if ts else "n/d",
            "score": round(_sentiment_titolo(titolo), 3),
        })
    if not recenti:
        return {"n": 0, "media": None, "label": "—", "articoli": []}
    media = round(sum(a["score"] for a in recenti) / len(recenti), 3)
    if media >= 0.15:
        label = "🟢 POSITIVO"
    elif media <= -0.15:
        label = "🔴 NEGATIVO"
    else:
        label = "⚪ NEUTRO"
    return {"n": len(recenti), "media": media, "label": label, "articoli": recenti}


def _fmt_levels(levels):
    return "  ·  ".join(f"€{x:g}" for x in levels) if levels else "—"


# ═══════════════════════════════════════════════════════════════════════
#  SEGNALI (segnali.db) e PORTAFOGLIO (portafoglio.json)
# ═══════════════════════════════════════════════════════════════════════

def _to_float(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    m = re.search(r"-?\d+[.,]?\d*", str(val).replace(",", "."))
    return float(m.group()) if m else None


@st.cache_data(ttl=120, show_spinner=False)
def carica_segnali() -> pd.DataFrame:
    import sqlite3
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM segnali ORDER BY data_segnale DESC", conn)
        conn.close()
    except Exception as e:
        st.error(f"Errore lettura segnali.db: {e}")
        return pd.DataFrame()
    for col in ("confidence_score", "convexity", "prezzo", "score_vader"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def carica_portafoglio() -> list:
    if os.path.exists(PORTFOLIO_PATH):
        try:
            with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def salva_portafoglio(posizioni: list):
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(posizioni, f, indent=2, ensure_ascii=False)


def carica_guadagni() -> list:
    """Registro delle vendite chiuse con guadagno realizzato."""
    if os.path.exists(GUADAGNI_PATH):
        try:
            with open(GUADAGNI_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def salva_guadagni(lst: list):
    with open(GUADAGNI_PATH, "w", encoding="utf-8") as f:
        json.dump(lst, f, indent=2, ensure_ascii=False)


def _fmt_rsi(v):
    if v is None:
        return "—"
    tag = " 🔴" if v >= 70 else (" 🟢" if v <= 30 else "")
    return f"{v}{tag}"


def _fmt_stoch(k, d):
    if k is None:
        return "—"
    tag = " 🔴" if k >= 80 else (" 🟢" if k <= 20 else "")
    return f"{k}/{d if d is not None else '—'}{tag}"


# ═══════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════

st.title("📈 Swing Trading Dashboard")
df_seg = carica_segnali()

c1, c2, c3 = st.columns([3, 2, 1])
with c1:
    if not df_seg.empty:
        st.caption(f"🗂️ Ultimo run segnali: **{df_seg['data_segnale'].max()}**  ·  "
                   f"{len(df_seg)} segnali in archivio")
    else:
        st.caption("⚠️ Nessun segnale trovato in segnali.db")
with c2:
    st.caption("📡 Fonte dati di mercato: **Yahoo Finance** · news ≤ 24h")
with c3:
    if st.button("🔄 Aggiorna", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

tab_opp, tab_port = st.tabs(["🔮 Opportunità", "💼 Portafoglio"])


# ═══════════════════════════════════════════════════════════════════════
#  TAB 1 — OPPORTUNITÀ
# ═══════════════════════════════════════════════════════════════════════

with tab_opp:
    if df_seg.empty:
        st.info("Nessun segnale disponibile. Esegui l'algoritmo per popolare segnali.db.")
    else:
        st.subheader("Filtri")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            tipi = sorted(df_seg["segnale"].dropna().unique().tolist())
            tipi_sel = st.multiselect("Tipo segnale", tipi, default=tipi)
        with f2:
            conf_min = st.slider("Confidence minima", 0, 100, 0, 5)
        with f3:
            regimi = sorted(df_seg["regime"].dropna().unique().tolist()) if "regime" in df_seg else []
            reg_sel = st.multiselect("Regime", regimi, default=regimi)
        with f4:
            max_live = st.number_input("Max titoli analizzati live", 5, 60, 15, 5)

        solo_ultimo = st.checkbox("Solo ultimo run", value=True)

        d = df_seg.copy()
        if solo_ultimo:
            d = d[d["data_segnale"] == d["data_segnale"].max()]
        if tipi_sel:
            d = d[d["segnale"].isin(tipi_sel)]
        if regimi and reg_sel:
            d = d[d["regime"].isin(reg_sel)]
        d = d[d["confidence_score"].fillna(0) >= conf_min]
        # un solo segnale per ticker (il più recente), ordinati per confidence
        d = d.drop_duplicates(subset=["ticker"], keep="first")
        d = d.sort_values("confidence_score", ascending=False, na_position="last")
        d = d.head(int(max_live))

        st.subheader(f"Opportunità selezionate ({len(d)})")

        if d.empty:
            st.warning("Nessuna opportunità con i filtri correnti.")
        else:
            with st.spinner(f"Aggiorno dati di mercato live per {len(d)} titoli da Yahoo Finance…"):
                righe = []
                for _, r in d.iterrows():
                    tk = r["ticker"]
                    m = metriche_mercato(tk)
                    fz = fondamentali(tk)
                    righe.append({
                        "Ticker": tk,
                        "Nome": fz["nome"] or tk,
                        "Settore": fz["settore"] or "—",
                        "Segnale": r.get("segnale", ""),
                        "Confidence": r.get("confidence_score"),
                        "Prezzo attuale €": m["prezzo"],
                        "Primo supporto €": m["sup1"],
                        "Prima resistenza €": m["res1"],
                        "Bollinger inf €": m["bb_inf"],
                        "Bollinger sup €": m["bb_sup"],
                        "Bollinger %B": m["bb_pct"],
                        "Posizione BB": m["bb_pos"],
                        "RSI": _fmt_rsi(m["rsi"]),
                        "Stocastico %K/%D": _fmt_stoch(m["stoch_k"], m["stoch_d"]),
                        "P/E": fz["pe"],
                        "P/B": fz["pb"],
                    })
            vista = pd.DataFrame(righe)
            st.dataframe(
                vista,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0, max_value=100, format="%d"),
                },
            )
            st.caption(
                "Confidence 0–100 dall'algoritmo. Nome, settore, prezzo, RSI, Stocastico, "
                "supporto/resistenza, Bollinger, P/E e P/B aggiornati live da Yahoo Finance. "
                "RSI 🔴≥70 / 🟢≤30; Stocastico 🔴≥80 / 🟢≤20. Bollinger (SMA20 ±2σ): %B = posizione "
                "nella banda (0=inf, 100=sup); Posizione BB 🔴 sopra / 🟢 sotto / ⚪ dentro. "
                "Supporto/resistenza = primo livello per vicinanza."
            )


# ═══════════════════════════════════════════════════════════════════════
#  TAB 2 — PORTAFOGLIO
# ═══════════════════════════════════════════════════════════════════════

with tab_port:
    if SOLO_VISUALIZZAZIONE:
        st.subheader("Le tue posizioni")
        st.info("👁️ **Modalità sola visualizzazione** — il portafoglio è gestito dal "
                "proprietario. Qui puoi consultare posizioni, indicatori, news e guadagni.")
    else:
        st.subheader("Le tue posizioni")
        st.caption("Inserisci l'**ISIN** (consigliato, es. IT0003856405) e/o il ticker/nome. "
                   "Il simbolo Yahoo ufficiale viene risolto automaticamente. "
                   "Per **chiudere** una posizione, scrivi il **Prezzo di vendita** e premi "
                   "«Salva»: il guadagno viene calcolato, registrato nei «Guadagni» del mese "
                   "e il titolo rimosso dalle posizioni attive.")

        posizioni = carica_portafoglio()
        df_pos = pd.DataFrame(posizioni) if posizioni else pd.DataFrame(
            columns=["isin", "ticker", "azioni", "prezzo_carico"])
        for c in ("isin", "ticker", "azioni", "prezzo_carico"):
            if c not in df_pos.columns:
                df_pos[c] = None
        df_pos["prezzo_vendita"] = None   # colonna di chiusura (sempre vuota all'apertura)

        edited = st.data_editor(
            df_pos[["isin", "ticker", "azioni", "prezzo_carico", "prezzo_vendita"]],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "isin": st.column_config.TextColumn("ISIN", help="Codice ISIN (es. IT0003856405). Risolve il simbolo ufficiale."),
                "ticker": st.column_config.TextColumn("Ticker / Nome", help="Opzionale se hai messo l'ISIN. Es. LDO.MI o 'Leonardo'."),
                "azioni": st.column_config.NumberColumn("N. azioni", min_value=0, step=1, format="%g"),
                "prezzo_carico": st.column_config.NumberColumn("Prezzo di carico €", min_value=0.0, format="%g"),
                "prezzo_vendita": st.column_config.NumberColumn(
                    "Prezzo di vendita €", min_value=0.0, format="%g",
                    help="Compila SOLO per chiudere la posizione: registra il guadagno e rimuove il titolo."),
            },
            key="editor_portafoglio",
        )

        cc1, cc2 = st.columns([1, 4])
        with cc1:
            if st.button("💾 Salva portafoglio", use_container_width=True):
                nuovi, chiusi = [], []
                oggi = datetime.now().strftime("%Y-%m-%d")
                for _, r in edited.iterrows():
                    isin_v = str(r["isin"]).strip().upper() if pd.notna(r["isin"]) else ""
                    tick_v = str(r["ticker"]).strip() if pd.notna(r["ticker"]) else ""
                    if not (isin_v or tick_v):
                        continue
                    azioni = float(r["azioni"]) if pd.notna(r["azioni"]) else 0.0
                    carico = float(r["prezzo_carico"]) if pd.notna(r["prezzo_carico"]) else 0.0
                    vendita = float(r["prezzo_vendita"]) if pd.notna(r["prezzo_vendita"]) else 0.0

                    if vendita > 0:
                        # ── Chiusura posizione: calcolo guadagno realizzato ──────
                        guadagno = round(azioni * (vendita - carico), 2)
                        chiusi.append({
                            "data_chiusura": oggi,
                            "mese": oggi[:7],                       # YYYY-MM
                            "isin": isin_v,
                            "ticker": tick_v,
                            "azioni": azioni,
                            "prezzo_carico": carico,
                            "prezzo_vendita": vendita,
                            "guadagno_eur": guadagno,
                        })
                    else:
                        # ── Posizione ancora attiva ──────────────────────────────
                        nuovi.append({
                            "isin": isin_v, "ticker": tick_v,
                            "azioni": azioni, "prezzo_carico": carico,
                        })

                salva_portafoglio(nuovi)
                if chiusi:
                    salva_guadagni(carica_guadagni() + chiusi)
                msg = f"Salvate {len(nuovi)} posizioni attive."
                if chiusi:
                    tot_g = sum(c["guadagno_eur"] for c in chiusi)
                    msg += f" Chiuse {len(chiusi)} posizioni — guadagno realizzato €{tot_g:,.2f}."
                st.success(msg)
                st.cache_data.clear()
                st.rerun()

    st.divider()

    posizioni = carica_portafoglio()
    if not posizioni:
        st.info("Nessuna posizione salvata. Aggiungi le tue azioni e premi «Salva».")
    else:
        with st.spinner("Risolvo i simboli (ISIN→Yahoo), aggiorno prezzi, indicatori e news (≤24h)…"):
            righe, news_per_ticker = [], {}
            tot_costo = tot_valore = 0.0
            non_risolti = []
            for p in posizioni:
                isin_v = p.get("isin", "") or ""
                tick_v = p.get("ticker", "") or ""
                etichetta = tick_v or isin_v
                azioni = float(p.get("azioni") or 0)
                carico = float(p.get("prezzo_carico") or 0)

                # ── Risoluzione simbolo ufficiale (ISIN preferito) ──────────
                ris = risolvi_simbolo(isin_v, tick_v)
                sym = ris["simbolo"]
                if sym is None:
                    non_risolti.append(etichetta)
                    news_per_ticker[etichetta] = {"n": 0, "media": None, "label": "—", "articoli": []}
                    costo = azioni * carico
                    tot_costo += costo
                    righe.append({
                        "Titolo": etichetta, "Simbolo Yahoo": "⚠️ non risolto",
                        "Azioni": azioni, "Carico €": round(carico, 2),
                        "Prezzo attuale €": None,
                        "RSI": "—", "Stocastico %K/%D": "—",
                        "Primo supporto €": None, "Prima resistenza €": None,
                        "News 24h": "—",
                    })
                    continue

                m = metriche_mercato(sym)
                ns = news_sentiment(sym, ore=24)
                news_per_ticker[f"{etichetta} ({sym})"] = ns
                attuale = m["prezzo"]

                costo = azioni * carico
                valore = azioni * attuale if attuale else None
                tot_costo += costo
                if valore is not None:
                    tot_valore += valore

                news_lbl = ns["label"]
                if ns["n"]:
                    news_lbl += f" {ns['media']:+.2f} ({ns['n']})"

                righe.append({
                    "Titolo": etichetta,
                    "Simbolo Yahoo": sym,
                    "Azioni": azioni,
                    "Carico €": round(carico, 2),
                    "Prezzo attuale €": round(attuale, 2) if attuale else None,
                    "RSI": _fmt_rsi(m["rsi"]),
                    "Stocastico %K/%D": _fmt_stoch(m["stoch_k"], m["stoch_d"]),
                    "Primo supporto €": m["sup1"],
                    "Prima resistenza €": m["res1"],
                    "News 24h": news_lbl,
                })

        if non_risolti:
            st.warning("⚠️ Simbolo non risolto per: " + ", ".join(non_risolti) +
                       ". Inserisci l'ISIN o il ticker Yahoo corretto (es. LDO.MI per Leonardo).")

        mm1, mm2, mm3 = st.columns(3)
        mm1.metric("Costo totale", f"€{tot_costo:,.2f}")
        mm2.metric("Valore attuale", f"€{tot_valore:,.2f}")
        mm3.metric("Posizioni attive", f"{len(posizioni)}")

        df_view = pd.DataFrame(righe)
        st.dataframe(df_view, use_container_width=True, hide_index=True)
        st.caption(
            "Prezzo, RSI, Stocastico, supporto/resistenza da Yahoo Finance. "
            "News 24h: sentiment medio finance-aware (VADER + lessico finanziario) "
            "sulle notizie ufficiali Yahoo delle ultime 24 ore. 🟢≥+0.15 / 🔴≤−0.15."
        )

        # ── Dettaglio news per titolo (ultime 24h) ─────────────────────
        st.markdown("#### 📰 News & sentiment (ultime 24h)")
        for tk, ns in news_per_ticker.items():
            with st.expander(f"{tk} — {ns['label']} "
                             f"{'(' + str(ns['n']) + ' notizie)' if ns['n'] else '(nessuna news 24h)'}"):
                if not ns["articoli"]:
                    st.write("Nessuna notizia nelle ultime 24 ore.")
                for a in ns["articoli"]:
                    seg = "🟢" if a["score"] >= 0.15 else ("🔴" if a["score"] <= -0.15 else "⚪")
                    titolo_md = f"[{a['titolo']}]({a['link']})" if a["link"] else a["titolo"]
                    st.markdown(f"{seg} **{a['score']:+.2f}** · {a['quando']} · "
                                f"_{a['publisher']}_ — {titolo_md}")

    # ═══════════════════════════════════════════════════════════════════
    #  GUADAGNI REALIZZATI (per mese)
    # ═══════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("💰 Guadagni")
    guadagni = carica_guadagni()
    if not guadagni:
        st.info("Nessuna vendita registrata. Chiudi una posizione (colonna «Prezzo di "
                "vendita» + «Salva») per registrare qui il guadagno.")
    else:
        dfg = pd.DataFrame(guadagni)
        dfg["guadagno_eur"] = pd.to_numeric(dfg["guadagno_eur"], errors="coerce")
        if "mese" not in dfg.columns:
            dfg["mese"] = dfg["data_chiusura"].str[:7]

        tot_realizzato = float(dfg["guadagno_eur"].sum())
        mese_corrente = datetime.now().strftime("%Y-%m")
        tot_mese = float(dfg[dfg["mese"] == mese_corrente]["guadagno_eur"].sum())
        g1, g2, g3 = st.columns(3)
        g1.metric("Guadagno totale realizzato", f"€{tot_realizzato:,.2f}")
        g2.metric(f"Guadagno mese {mese_corrente}", f"€{tot_mese:,.2f}")
        g3.metric("Operazioni chiuse", f"{len(dfg)}")

        # ── Annulla ultima chiusura (solo proprietario) ────────────────
        if not SOLO_VISUALIZZAZIONE:
            ultimo = guadagni[-1]
            u1, u2 = st.columns([1, 3])
            with u1:
                if st.button("↩️ Annulla ultima chiusura", use_container_width=True):
                    g_list = carica_guadagni()
                    if g_list:
                        rip = g_list.pop()                       # rimuove l'ultima registrata
                        salva_guadagni(g_list)
                        port = carica_portafoglio()
                        port.append({                            # ripristina come posizione attiva
                            "isin": rip.get("isin", ""),
                            "ticker": rip.get("ticker", ""),
                            "azioni": rip.get("azioni", 0.0),
                            "prezzo_carico": rip.get("prezzo_carico", 0.0),
                        })
                        salva_portafoglio(port)
                        st.success(f"Ripristinata «{rip.get('ticker') or rip.get('isin')}» tra le "
                                   f"posizioni attive (annullato guadagno €{rip.get('guadagno_eur', 0):,.2f}).")
                        st.cache_data.clear()
                        st.rerun()
            with u2:
                st.caption(f"L'ultima chiusura registrata è «{ultimo.get('ticker') or ultimo.get('isin')}» "
                           f"del {ultimo.get('data_chiusura', 'n/d')} "
                           f"(guadagno €{float(ultimo.get('guadagno_eur', 0)):,.2f}). "
                           f"Annullandola torna tra le posizioni attive.")

        # Riepilogo per mese
        per_mese = (dfg.groupby("mese", as_index=False)
                       .agg(Guadagno=("guadagno_eur", "sum"),
                            Operazioni=("guadagno_eur", "count"))
                       .sort_values("mese", ascending=False))
        per_mese["Guadagno"] = per_mese["Guadagno"].round(2)
        per_mese = per_mese.rename(columns={"mese": "Mese"})
        st.markdown("**Guadagni per mese**")
        st.dataframe(per_mese, use_container_width=True, hide_index=True)

        # Dettaglio operazioni chiuse
        with st.expander("📋 Dettaglio operazioni chiuse"):
            dett = dfg.copy()
            for c in ("prezzo_carico", "prezzo_vendita", "guadagno_eur"):
                if c in dett.columns:
                    dett[c] = pd.to_numeric(dett[c], errors="coerce").round(2)
            cols = [c for c in ["data_chiusura", "ticker", "isin", "azioni",
                                "prezzo_carico", "prezzo_vendita", "guadagno_eur"] if c in dett.columns]
            dett = dett[cols].rename(columns={
                "data_chiusura": "Data", "ticker": "Titolo", "isin": "ISIN",
                "azioni": "Azioni", "prezzo_carico": "Carico €",
                "prezzo_vendita": "Vendita €", "guadagno_eur": "Guadagno €"})
            st.dataframe(dett.sort_values("Data", ascending=False),
                         use_container_width=True, hide_index=True)
