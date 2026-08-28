from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

FILE = Path(__file__).with_name("USAstocks.xlsx")
SP500 = "SPY"
NASDAQ100 = "QQQ"
HEDGE_TICKER = "QQQ"
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_HEDGE_MULTIPLIER = 2.5
CURRENCY = "$"
TZ = ZoneInfo("Europe/Madrid")
QUICK_PERIODS = {
    "1S": pd.DateOffset(weeks=1),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
}
MONTHS = {
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Ago": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12,
}
MONTH_NAMES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

st.set_page_config(
    page_title="Rentabilidad de la cartera",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root{color-scheme:light!important}
html,body,[data-testid="stAppViewContainer"],.stApp{background:#fff!important;color:#172033!important}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important}
.block-container{padding-top:1.2rem;padding-bottom:2.5rem;max-width:1450px}
h1{font-size:2rem!important;color:#132238!important;letter-spacing:-.03em}
[data-testid="stMetric"]{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:14px 16px;box-shadow:0 4px 14px rgba(15,23,42,.055)}
[data-testid="stMetricLabel"] p{color:#64748b!important}
[data-testid="stMetricValue"]{color:#172033!important;font-weight:750}
div[role="radiogroup"]{gap:.35rem;flex-wrap:wrap!important}
div[role="radiogroup"] label{background:#f1f5f9!important;border:1px solid #cbd5e1!important;border-radius:10px!important;padding:.34rem .72rem!important;min-width:54px!important;justify-content:center!important;color:#172033!important}
div[role="radiogroup"] label p,div[role="radiogroup"] label span{color:#172033!important;font-weight:700!important;opacity:1!important}
div[role="radiogroup"] label:has(input:checked){background:#143d59!important;border-color:#143d59!important}
div[role="radiogroup"] label:has(input:checked) p,div[role="radiogroup"] label:has(input:checked) span{color:#fff!important;font-weight:800!important}
.section{font-size:.82rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:750;margin:.8rem 0 .55rem}
.month-label{color:#64748b;font-size:.82rem;font-weight:750;margin:.15rem 0 .25rem}
.small-note{color:#64748b;font-size:.82rem}
@media(max-width:768px){.block-container{padding:1rem .75rem 2rem!important}h1{font-size:2rem!important;line-height:1.15!important}div[role="radiogroup"]{gap:.25rem!important}div[role="radiogroup"] label{min-width:48px!important;padding:.28rem .42rem!important}div[role="radiogroup"] label p{font-size:.82rem!important}}
</style>
""",
    unsafe_allow_html=True,
)


def load_trades(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(column).strip().upper() for column in df.columns]
    required = {"TICKER", "BUY DATE", "SELL DATE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(sorted(missing)))

    df["TICKER"] = df["TICKER"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    df["TICKER"] = df["TICKER"].replace({"SATS": "ECHO"})
    df["BUY DATE"] = pd.to_datetime(df["BUY DATE"], errors="coerce").dt.normalize()
    df["SELL DATE"] = pd.to_datetime(df["SELL DATE"], errors="coerce").dt.normalize()
    df = df[df["BUY DATE"].notna() & df["TICKER"].ne("")].copy()
    if df.empty:
        raise ValueError("El Excel no contiene operaciones válidas.")
    invalid = df["SELL DATE"].notna() & (df["SELL DATE"] < df["BUY DATE"])
    if invalid.any():
        raise ValueError("Hay alguna fecha de venta anterior a la compra.")
    df["TRADE ID"] = np.arange(len(df))
    return df.sort_values(["BUY DATE", "TICKER"]).reset_index(drop=True)


@st.cache_data(ttl=55, show_spinner=False)
def get_prices(tickers, start, end):
    raw = yf.download(
        list(tickers), start=start, end=end, auto_adjust=True,
        progress=False, threads=True, timeout=20,
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    return prices.sort_index().dropna(how="all")


def next_session(date, sessions):
    candidates = sessions[sessions >= pd.Timestamp(date)]
    return candidates[0] if len(candidates) else pd.NaT


def build_long_portfolio(trades, prices, initial_capital):
    sessions = prices.index
    events = trades.copy()
    events["BUY SESSION"] = events["BUY DATE"].map(lambda value: next_session(value, sessions))
    events["SELL SESSION"] = events["SELL DATE"].map(
        lambda value: pd.NaT if pd.isna(value) else next_session(value, sessions)
    )
    event_dates = sorted(set(events["BUY SESSION"].dropna()) | set(events["SELL SESSION"].dropna()))
    if not event_dates:
        raise ValueError("No se han encontrado fechas de operación válidas.")

    first_event = event_dates[0]
    holdings = {}
    cash = float(initial_capital)
    records = []
    ledger = {}

    for date in sessions[sessions >= first_event]:
        prices_today = prices.loc[date]
        if date in event_dates:
            sales = events[events["SELL SESSION"] == date]
            purchases = events[events["BUY SESSION"] == date]
            sale_proceeds = 0.0

            for _, trade in sales.iterrows():
                trade_id = int(trade["TRADE ID"])
                position = holdings.pop(trade_id, None)
                price = prices_today.get(trade["TICKER"], np.nan)
                if position and pd.notna(price):
                    exit_value = position["shares"] * price
                    sale_proceeds += exit_value
                    cash += exit_value
                    ledger[trade_id].update({
                        "EXIT VALUE": exit_value,
                        "CURRENT VALUE": exit_value,
                        "P&L MONEY": exit_value - position["entry_value"],
                    })

            valid_buys = []
            for _, trade in purchases.iterrows():
                price = prices_today.get(trade["TICKER"], np.nan)
                if pd.notna(price) and price > 0:
                    valid_buys.append(trade)

            if valid_buys:
                investment_pool = cash if date == first_event else min(sale_proceeds, cash)
                amount_per_trade = investment_pool / len(valid_buys)
                for trade in valid_buys:
                    trade_id = int(trade["TRADE ID"])
                    ticker = trade["TICKER"]
                    price = prices_today[ticker]
                    shares = amount_per_trade / price
                    holdings[trade_id] = {
                        "ticker": ticker,
                        "shares": shares,
                        "entry_value": amount_per_trade,
                    }
                    cash -= amount_per_trade
                    ledger[trade_id] = {
                        "ENTRY VALUE": amount_per_trade,
                        "EXIT VALUE": np.nan,
                        "CURRENT VALUE": amount_per_trade,
                        "P&L MONEY": 0.0,
                    }

        invested = 0.0
        for trade_id, position in holdings.items():
            price = prices_today.get(position["ticker"], np.nan)
            if pd.notna(price):
                current_value = position["shares"] * price
                invested += current_value
                ledger[trade_id]["CURRENT VALUE"] = current_value
                ledger[trade_id]["P&L MONEY"] = current_value - position["entry_value"]

        records.append({"Fecha": date, "Patrimonio": cash + invested})

    portfolio = pd.DataFrame(records).set_index("Fecha")
    ledger_df = pd.DataFrame.from_dict(ledger, orient="index")
    ledger_df.index.name = "TRADE ID"
    return portfolio, ledger_df


def build_hedge(long_equity, prices, hedge_date, multiplier):
    """Abre una cobertura corta fija en QQQ y no la rebalancea después."""
    hedge_session = next_session(hedge_date, prices.index)
    if pd.isna(hedge_session) or hedge_session not in long_equity.index:
        raise ValueError("No se puede determinar la sesión de apertura de la cobertura.")

    base_value = float(long_equity.loc[hedge_session])
    entry_price = float(prices.loc[hedge_session, HEDGE_TICKER])
    notional = base_value * float(multiplier)
    short_shares = notional / entry_price

    qqq = prices[HEDGE_TICKER].reindex(long_equity.index).ffill()
    hedge_pnl = pd.Series(0.0, index=long_equity.index)
    active = hedge_pnl.index >= hedge_session
    hedge_pnl.loc[active] = (entry_price - qqq.loc[active]) * short_shares
    hedged_equity = long_equity + hedge_pnl

    details = {
        "session": hedge_session,
        "base_value": base_value,
        "entry_price": entry_price,
        "notional": notional,
        "short_shares": short_shares,
        "current_price": float(qqq.iloc[-1]),
        "current_liability": float(short_shares * qqq.iloc[-1]),
        "pnl": float(hedge_pnl.iloc[-1]),
        "return": float(hedge_pnl.iloc[-1] / notional) if notional else np.nan,
    }
    return hedged_equity.rename("Cartera con cobertura"), hedge_pnl, details


def build_buy_hold(prices, ticker, start_date, capital):
    series = prices[ticker].dropna().loc[start_date:]
    return series * (capital / series.iloc[0])


def quick_range(period, today):
    if period == "YTD":
        return pd.Timestamp(today.year, 1, 1), today
    return (today - QUICK_PERIODS[period]).normalize(), today


def month_range(month_number, year, today):
    start = pd.Timestamp(year, month_number, 1)
    end = start + pd.offsets.MonthEnd(1)
    return start, min(end, today)


def select_period(values, start, end):
    chosen = values.loc[(values.index >= start) & (values.index <= end)].copy()
    if chosen.empty:
        raise ValueError("No hay sesiones disponibles para el periodo seleccionado.")
    return chosen


def calculate_stats(values):
    normalized = values / values.iloc[0]
    total_return = normalized.iloc[-1] - 1
    drawdown = normalized / normalized.cummax() - 1
    max_drawdown = drawdown.min()
    return {
        "return": total_return,
        "dd": max_drawdown,
        "ratio": total_return / abs(max_drawdown) if max_drawdown < 0 else np.nan,
    }


def beta_alpha(asset_values, benchmark_values):
    aligned = pd.concat([asset_values, benchmark_values], axis=1).dropna()
    returns = aligned.pct_change(fill_method=None).dropna()
    if len(returns) < 3 or returns.iloc[:, 1].var() == 0:
        return np.nan, np.nan
    asset_return = returns.iloc[:, 0]
    benchmark_return = returns.iloc[:, 1]
    beta = asset_return.cov(benchmark_return) / benchmark_return.var()
    alpha_daily = (asset_return - beta * benchmark_return).mean()
    alpha_annual = (1 + alpha_daily) ** 252 - 1
    return beta, alpha_annual


def pct(value):
    return "N/D" if pd.isna(value) else f"{value:+.2%}"


def beta_text(value):
    return "N/D" if pd.isna(value) else f"{value:.2f}"


def ratio(value):
    return "N/D" if pd.isna(value) or np.isinf(value) else f"{value:.2f}x"


def money(value, signed=False):
    if pd.isna(value):
        return "-"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{CURRENCY}{value:,.0f}"


def common_layout(fig, height, title):
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=15, t=85, b=35),
        title=dict(text=title, x=.01, font=dict(color="#172033", size=18)),
        paper_bgcolor="#fff", plot_bgcolor="#f8fafc",
        font=dict(color="#334155", size=13),
        xaxis=dict(showgrid=False, linecolor="#94a3b8", tickfont=dict(color="#334155", size=12), automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#334155", size=12), bgcolor="rgba(255,255,255,.85)"),
        hoverlabel=dict(bgcolor="#111827", bordercolor="#111827", font=dict(color="#fff", size=13)),
        hovermode="x unified",
    )


def choose_quick():
    st.session_state.active_selector = "quick"
    st.session_state.month_choice = None


def choose_month():
    st.session_state.active_selector = "month"
    st.session_state.quick_choice = None


now = datetime.now(TZ)
today = pd.Timestamp(now.date())
if "active_selector" not in st.session_state:
    st.session_state.active_selector = "quick"
if "quick_choice" not in st.session_state:
    st.session_state.quick_choice = "YTD"
if "month_choice" not in st.session_state:
    st.session_state.month_choice = None

st.title("Rentabilidad de la cartera")
capital_col, hedge_col, selector_col, refresh_col = st.columns([2.0, 1.7, 4.8, 1.5], vertical_alignment="bottom")
with capital_col:
    initial_capital = st.number_input(
        "Capital inicial a 1 de enero",
        min_value=1_000.0,
        value=DEFAULT_INITIAL_CAPITAL,
        step=10_000.0,
        format="%.0f",
    )
with hedge_col:
    hedge_multiplier = st.number_input(
        "Multiplicador cobertura",
        min_value=0.0,
        value=DEFAULT_HEDGE_MULTIPLIER,
        step=0.1,
        format="%.2f",
        help="Notional corto en QQQ = valor de mercado de la cartera el día de apertura × multiplicador.",
    )
with selector_col:
    st.radio(
        "Periodo rápido", ["1S", "1M", "3M", "6M", "YTD"],
        index=None, key="quick_choice", horizontal=True,
        label_visibility="collapsed", on_change=choose_quick,
    )
with refresh_col:
    if st.button("↻ Actualizar", type="primary", width="stretch"):
        st.cache_data.clear()
        st.rerun()

st.markdown('<div class="month-label">Seleccionar mes</div>', unsafe_allow_html=True)
st.radio(
    "Mes", list(MONTHS.keys()), index=None, key="month_choice",
    horizontal=True, label_visibility="collapsed", on_change=choose_month,
)

if not FILE.exists():
    st.error("No encuentro USAstocks.xlsx junto a app.py.")
    st.stop()

try:
    all_rows = load_trades(FILE)
    hedge_rows = all_rows[all_rows["TICKER"] == HEDGE_TICKER].copy()
    long_trades = all_rows[all_rows["TICKER"] != HEDGE_TICKER].copy()
    if hedge_rows.empty:
        raise ValueError("No encuentro la fila de cobertura QQQ en el Excel.")
    hedge_row = hedge_rows.iloc[-1]
    hedge_date = pd.Timestamp(hedge_row["BUY DATE"])

    tickers = tuple(sorted(set(long_trades["TICKER"]) | {SP500, NASDAQ100}))
    prices = get_prices(
        tickers,
        (all_rows["BUY DATE"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        (today + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
    )
    if prices.empty or SP500 not in prices.columns or NASDAQ100 not in prices.columns:
        st.error("No se han podido descargar los precios de SPY o QQQ.")
        st.stop()

    missing = sorted(set(long_trades["TICKER"]) - set(prices.columns))
    if missing:
        st.warning("Sin precios para: " + ", ".join(missing))
        long_trades = long_trades[~long_trades["TICKER"].isin(missing)].copy()

    portfolio, ledger = build_long_portfolio(long_trades, prices, initial_capital)
    long_equity = portfolio["Patrimonio"].rename("Cartera sin cobertura")
    hedged_equity, hedge_pnl, hedge_details = build_hedge(
        long_equity, prices, hedge_date, hedge_multiplier
    )
    spy_equity = build_buy_hold(prices, SP500, portfolio.index[0], initial_capital).rename("S&P 500")
    qqq_equity = build_buy_hold(prices, NASDAQ100, portfolio.index[0], initial_capital).rename("Nasdaq 100")
    equity = pd.concat([long_equity, hedged_equity, spy_equity, qqq_equity], axis=1).ffill().dropna()

    if st.session_state.active_selector == "month" and st.session_state.month_choice:
        month_number = MONTHS[st.session_state.month_choice]
        start, end = month_range(month_number, today.year, today)
        period_label = MONTH_NAMES[month_number]
    else:
        quick = st.session_state.quick_choice or "YTD"
        start, end = quick_range(quick, today)
        period_label = quick

    selected = select_period(equity, start, end)
    long_stats = calculate_stats(selected["Cartera sin cobertura"])
    hedge_stats = calculate_stats(selected["Cartera con cobertura"])
    spy_stats = calculate_stats(selected["S&P 500"])
    drawdown = selected / selected.cummax() - 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=selected.index, y=selected["Cartera sin cobertura"],
        name="Cartera sin cobertura", mode="lines",
        line=dict(color="#087f8c", width=3),
        hovertemplate=f"%{{x|%d/%m/%Y}}<br><b>{CURRENCY}%{{y:,.0f}}</b><extra>Sin cobertura</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=selected.index, y=selected["Cartera con cobertura"],
        name=f"Cartera con cobertura ×{hedge_multiplier:g}", mode="lines",
        line=dict(color="#8b1e3f", width=3),
        hovertemplate=f"%{{x|%d/%m/%Y}}<br><b>{CURRENCY}%{{y:,.0f}}</b><extra>Con cobertura</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=selected.index, y=selected["S&P 500"], name="S&P 500 (SPY)", mode="lines",
        line=dict(color="#e07a3f", width=2.3),
        hovertemplate=f"%{{x|%d/%m/%Y}}<br><b>{CURRENCY}%{{y:,.0f}}</b><extra>S&P 500</extra>",
    ))
    common_layout(fig, 570, f"Valor de la cartera · {period_label} · Última sesión: {selected.index[-1]:%d/%m/%Y}")
    fig.update_yaxes(title="Valor de la cartera", tickprefix=CURRENCY, tickformat=",.0f", gridcolor="#dbe3eb", automargin=True)
    st.plotly_chart(fig, width="stretch", config={"displaylogo":False, "displayModeBar":False, "responsive":True})

    st.markdown('<div class="section">Situación actual</div>', unsafe_allow_html=True)
    initial_period_value = selected["Cartera sin cobertura"].iloc[0]
    current_long = selected["Cartera sin cobertura"].iloc[-1]
    current_hedged = selected["Cartera con cobertura"].iloc[-1]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric(f"Cantidad inicial {period_label}", money(initial_period_value))
    p2.metric("Actual sin cobertura", money(current_long))
    p3.metric("Actual con cobertura", money(current_hedged))
    p4.metric("P&L cobertura", money(hedge_details["pnl"], signed=True))

    st.markdown('<div class="section">Rentabilidad y riesgo</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**Cartera sin cobertura**")
        a, b, c = st.columns(3)
        a.metric("Rentabilidad", pct(long_stats["return"]))
        b.metric("Drawdown", pct(long_stats["dd"]))
        c.metric("Rent./DD", ratio(long_stats["ratio"]))
    with r2:
        st.markdown(f"**Cartera con cobertura ×{hedge_multiplier:g}**")
        a, b, c = st.columns(3)
        a.metric("Rentabilidad", pct(hedge_stats["return"]))
        b.metric("Drawdown", pct(hedge_stats["dd"]))
        c.metric("Rent./DD", ratio(hedge_stats["ratio"]))
    with r3:
        st.markdown("**S&P 500**")
        a, b, c = st.columns(3)
        a.metric("Rentabilidad", pct(spy_stats["return"]))
        b.metric("Drawdown", pct(spy_stats["dd"]))
        c.metric("Rent./DD", ratio(spy_stats["ratio"]))

    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown["Cartera sin cobertura"] * 100,
        name="Sin cobertura", mode="lines",
        line=dict(color="#087f8c", width=2.6),
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>Sin cobertura</extra>",
    ))
    dd_fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown["Cartera con cobertura"] * 100,
        name=f"Con cobertura ×{hedge_multiplier:g}", mode="lines",
        line=dict(color="#8b1e3f", width=2.8), fill="tozeroy",
        fillcolor="rgba(139,30,63,.12)",
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>Con cobertura</extra>",
    ))
    dd_fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown["S&P 500"] * 100,
        name="S&P 500", mode="lines",
        line=dict(color="#e07a3f", width=2.1),
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>S&P 500</extra>",
    ))
    dd_fig.add_hline(y=0, line_width=1, line_color="#94a3b8")
    common_layout(dd_fig, 430, f"Curva de drawdown · {period_label}")
    dd_fig.update_yaxes(title="Drawdown", ticksuffix="%", gridcolor="#dbe3eb", rangemode="tozero")
    st.plotly_chart(dd_fig, width="stretch", config={"displaylogo":False, "displayModeBar":False, "responsive":True})

    long_beta, long_alpha = beta_alpha(selected["Cartera sin cobertura"], selected["S&P 500"])
    hedged_beta, hedged_alpha = beta_alpha(selected["Cartera con cobertura"], selected["S&P 500"])
    qqq_beta, qqq_alpha = beta_alpha(selected["Nasdaq 100"], selected["S&P 500"])

    st.markdown('<div class="section">Beta y alpha anualizada frente al S&P 500</div>', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Beta cartera", beta_text(long_beta), delta=f"Alpha {pct(long_alpha)}", delta_color="off")
    b2.metric("Beta cartera cubierta", beta_text(hedged_beta), delta=f"Alpha {pct(hedged_alpha)}", delta_color="off")
    b3.metric("Beta S&P 500", "1.00", delta="Alpha +0.00%", delta_color="off")
    b4.metric("Beta Nasdaq 100", beta_text(qqq_beta), delta=f"Alpha {pct(qqq_alpha)}", delta_color="off")

    st.markdown('<div class="section">Detalles de la cobertura</div>', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Apertura", hedge_details["session"].strftime("%d/%m/%Y"))
    h2.metric("Valor cartera al abrir", money(hedge_details["base_value"]))
    h3.metric("Notional corto QQQ", money(hedge_details["notional"]))
    h4.metric("Resultado cobertura", money(hedge_details["pnl"], signed=True))

    year_start = pd.Timestamp(today.year, 1, 1)
    history = long_trades[(long_trades["BUY DATE"] >= year_start) & (long_trades["BUY DATE"] <= today)].copy()
    history = history.merge(ledger, how="left", left_on="TRADE ID", right_index=True)
    history["ESTADO"] = np.where(history["SELL DATE"].notna() & (history["SELL DATE"] <= today), "Cerrada", "Abierta")
    history["TIPO"] = "Inversión"
    history["FINAL"] = history["SELL DATE"].where(history["ESTADO"].eq("Cerrada"), today)
    history["DURACION"] = (history["FINAL"] - history["BUY DATE"]).dt.days
    history["RENTABILIDAD"] = history["P&L MONEY"] / history["ENTRY VALUE"]

    hedge_history = pd.DataFrame([{
        "TICKER": HEDGE_TICKER,
        "COMPANY": "Cobertura Nasdaq 100",
        "GICS SECTOR": "Cobertura",
        "GICS SUB-INDUSTRY": "Posición corta",
        "BUY DATE": hedge_details["session"],
        "SELL DATE": pd.NaT,
        "ESTADO": "Abierta",
        "TIPO": f"Cobertura corta ×{hedge_multiplier:g}",
        "DURACION": (today - hedge_details["session"]).days,
        "ENTRY VALUE": hedge_details["notional"],
        "CURRENT VALUE": hedge_details["current_liability"],
        "P&L MONEY": hedge_details["pnl"],
        "RENTABILIDAD": hedge_details["return"],
    }])
    history = pd.concat([history, hedge_history], ignore_index=True, sort=False)
    open_count = int(history["ESTADO"].eq("Abierta").sum())
    closed_count = int(history["ESTADO"].eq("Cerrada").sum())

    st.subheader(f"Histórico de operaciones {today.year}")
    st.caption(f"{len(history)} operaciones · {open_count} abiertas · {closed_count} cerradas · incluye la cobertura")
    wanted = [
        "TIPO", "TICKER", "COMPANY", "GICS SECTOR", "GICS INDUSTRY", "GICS SUB-INDUSTRY",
        "BUY DATE", "SELL DATE", "ESTADO", "DURACION", "ENTRY VALUE",
        "CURRENT VALUE", "P&L MONEY", "RENTABILIDAD",
    ]
    operations = history[[column for column in wanted if column in history.columns]].copy()
    operations["ORDEN ESTADO"] = operations["ESTADO"].map({"Abierta":0, "Cerrada":1})
    operations = operations.sort_values(["ORDEN ESTADO", "BUY DATE", "TICKER"], ascending=[True, False, True]).drop(columns="ORDEN ESTADO")
    operations["BUY DATE"] = pd.to_datetime(operations["BUY DATE"]).dt.strftime("%d/%m/%Y")
    operations["SELL DATE"] = pd.to_datetime(operations["SELL DATE"]).dt.strftime("%d/%m/%Y").fillna("-")
    operations["ESTADO"] = operations["ESTADO"].map({"Abierta":"🟢 Abierta", "Cerrada":"🔴 Cerrada"})
    operations = operations.rename(columns={
        "TIPO":"Tipo", "TICKER":"Ticker", "COMPANY":"Compañía",
        "GICS SECTOR":"Sector GICS", "GICS INDUSTRY":"Industria GICS",
        "GICS SUB-INDUSTRY":"Subindustria GICS", "BUY DATE":"Fecha de compra",
        "SELL DATE":"Fecha de cierre", "ESTADO":"Estado", "DURACION":"Duración (días)",
        "ENTRY VALUE":"Importe inicial", "CURRENT VALUE":"Importe final/actual",
        "P&L MONEY":"Ganancia / pérdida", "RENTABILIDAD":"Rentabilidad",
    })

    def pnl_style(value):
        if pd.isna(value): return "font-weight:700;color:#64748b"
        if value > 0: return "font-weight:800;color:#0a8f55"
        if value < 0: return "font-weight:800;color:#d93025"
        return "font-weight:800;color:#475569"

    styled = (
        operations.style
        .map(pnl_style, subset=["Rentabilidad", "Ganancia / pérdida"])
        .format({
            "Rentabilidad": lambda value: "-" if pd.isna(value) else f"{value:+.2%}",
            "Ganancia / pérdida": lambda value: money(value, signed=True),
            "Importe inicial": lambda value: money(value),
            "Importe final/actual": lambda value: money(value),
            "Duración (días)": lambda value: "-" if pd.isna(value) else f"{int(value)} días",
        })
    )
    st.dataframe(styled, width="stretch", hide_index=True)

    st.caption(
        "La cobertura se modela como una posición corta fija en QQQ, abierta en la primera sesión "
        "disponible desde la fecha indicada en el Excel. El notional inicial es el valor total de "
        "la cartera en esa sesión multiplicado por el factor seleccionado. No se rebalancea después. "
        "No se incluyen comisiones, financiación, dividendos debitados ni impuestos."
    )

except Exception as exc:
    st.error(f"No se ha podido generar el dashboard: {exc}")
    st.exception(exc)
