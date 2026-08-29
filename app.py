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
HEDGE_TICKER = "QQQ"
DEFAULT_CAPITAL = 100_000.0
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
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12,
}
MONTH_NAMES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

st.set_page_config(page_title="Rentabilidad de la cartera", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{color-scheme:light!important}html,body,[data-testid="stAppViewContainer"],.stApp{background:#fff!important;color:#172033!important}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important}.block-container{padding-top:1.2rem;padding-bottom:2.5rem;max-width:1450px}
h1{font-size:2rem!important;color:#132238!important;letter-spacing:-.03em}[data-testid="stMetric"]{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:14px 16px;box-shadow:0 4px 14px rgba(15,23,42,.055)}
[data-testid="stMetricLabel"] p{color:#64748b!important}[data-testid="stMetricValue"]{color:#172033!important;font-weight:750}
div[role="radiogroup"]{gap:.35rem;flex-wrap:wrap!important}div[role="radiogroup"] label{background:#f1f5f9!important;border:1px solid #cbd5e1!important;border-radius:10px!important;padding:.34rem .72rem!important;min-width:54px!important;justify-content:center!important;color:#172033!important}
div[role="radiogroup"] label p,div[role="radiogroup"] label span{color:#172033!important;font-weight:700!important;opacity:1!important}div[role="radiogroup"] label:has(input:checked){background:#143d59!important;border-color:#143d59!important}div[role="radiogroup"] label:has(input:checked) p,div[role="radiogroup"] label:has(input:checked) span{color:#fff!important;font-weight:800!important}
.section{font-size:.82rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:750;margin:.8rem 0 .55rem}.month-label{color:#64748b;font-size:.82rem;font-weight:750;margin:.15rem 0 .25rem}
@media(max-width:768px){.block-container{padding:1rem .75rem 2rem!important}h1{font-size:2rem!important;line-height:1.15!important}div[role="radiogroup"]{gap:.25rem!important}div[role="radiogroup"] label{min-width:48px!important;padding:.28rem .42rem!important}div[role="radiogroup"] label p{font-size:.82rem!important}}
</style>
""", unsafe_allow_html=True)


def load_workbook_data(path):
    raw = pd.read_excel(path, header=None, engine="openpyxl")
    header_rows = raw.index[raw.iloc[:, 0].astype(str).str.strip().str.upper().eq("TICKER")].tolist()
    if not header_rows:
        raise ValueError("No encuentro la cabecera principal del Excel.")

    first = header_rows[0]
    second = header_rows[1] if len(header_rows) > 1 else len(raw)
    main = raw.iloc[first + 1:second, :6].copy()
    main.columns = ["TICKER", "COMPANY", "GICS SECTOR", "GICS SUB-INDUSTRY", "BUY DATE", "SELL DATE"]
    main = main[main["TICKER"].notna()].copy()
    main["TICKER"] = main["TICKER"].astype(str).str.strip().str.upper().replace({"SATS": "ECHO"})
    main["BUY DATE"] = pd.to_datetime(main["BUY DATE"], errors="coerce").dt.normalize()
    main["SELL DATE"] = pd.to_datetime(main["SELL DATE"], errors="coerce").dt.normalize()
    main = main[main["BUY DATE"].notna()].copy().reset_index(drop=True)
    main["TRADE ID"] = np.arange(len(main))

    hedges = pd.DataFrame(columns=["TICKER", "COMPANY", "OPEN DATE", "CLOSE DATE"])
    if len(header_rows) > 1:
        hedge = raw.iloc[second + 1:, :4].copy()
        # En la sección hedge: SELL DATE abre el corto y BUY DATE lo cierra.
        hedge.columns = ["TICKER", "COMPANY", "OPEN DATE", "CLOSE DATE"]
        hedge = hedge[hedge["TICKER"].notna()].copy()
        hedge["TICKER"] = hedge["TICKER"].astype(str).str.strip().str.upper()
        hedge["OPEN DATE"] = pd.to_datetime(hedge["OPEN DATE"], errors="coerce").dt.normalize()
        hedge["CLOSE DATE"] = pd.to_datetime(hedge["CLOSE DATE"], errors="coerce").dt.normalize()
        hedges = hedge[hedge["TICKER"].eq(HEDGE_TICKER) & hedge["OPEN DATE"].notna()].reset_index(drop=True)
        hedges["HEDGE ID"] = np.arange(len(hedges))

    if main.empty:
        raise ValueError("No hay operaciones de cartera válidas.")
    if hedges.empty:
        raise ValueError("No hay operaciones de cobertura QQQ válidas en la segunda tabla.")
    return main, hedges


@st.cache_data(ttl=55, show_spinner=False)
def get_prices(tickers, start, end):
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False, threads=True, timeout=20)
    if raw.empty:
        return pd.DataFrame()
    prices = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    return prices.sort_index().dropna(how="all")


def next_session(date, sessions):
    candidates = sessions[sessions >= pd.Timestamp(date)]
    return candidates[0] if len(candidates) else pd.NaT


def build_long_portfolio(trades, prices, capital):
    sessions = prices.index
    events = trades.copy()
    events["BUY SESSION"] = events["BUY DATE"].map(lambda d: next_session(d, sessions))
    events["SELL SESSION"] = events["SELL DATE"].map(lambda d: pd.NaT if pd.isna(d) else next_session(d, sessions))
    event_dates = sorted(set(events["BUY SESSION"].dropna()) | set(events["SELL SESSION"].dropna()))
    first_event = event_dates[0]
    holdings, ledger, records = {}, {}, []
    cash = float(capital)

    for date in sessions[sessions >= first_event]:
        px = prices.loc[date]
        if date in event_dates:
            sales = events[events["SELL SESSION"].eq(date)]
            buys = events[events["BUY SESSION"].eq(date)]
            proceeds = 0.0
            for _, trade in sales.iterrows():
                trade_id = int(trade["TRADE ID"])
                pos = holdings.pop(trade_id, None)
                price = px.get(trade["TICKER"], np.nan)
                if pos and pd.notna(price):
                    exit_value = pos["shares"] * price
                    proceeds += exit_value
                    cash += exit_value
                    ledger[trade_id].update({"CURRENT VALUE": exit_value, "P&L MONEY": exit_value - pos["entry_value"]})

            valid_buys = [row for _, row in buys.iterrows() if pd.notna(px.get(row["TICKER"], np.nan)) and px[row["TICKER"]] > 0]
            if valid_buys:
                pool = cash if date == first_event else min(proceeds, cash)
                allocation = pool / len(valid_buys)
                for trade in valid_buys:
                    trade_id, ticker = int(trade["TRADE ID"]), trade["TICKER"]
                    shares = allocation / px[ticker]
                    holdings[trade_id] = {"ticker": ticker, "shares": shares, "entry_value": allocation}
                    cash -= allocation
                    ledger[trade_id] = {"ENTRY VALUE": allocation, "CURRENT VALUE": allocation, "P&L MONEY": 0.0}

        invested = 0.0
        for trade_id, pos in holdings.items():
            price = px.get(pos["ticker"], np.nan)
            if pd.notna(price):
                value = pos["shares"] * price
                invested += value
                ledger[trade_id]["CURRENT VALUE"] = value
                ledger[trade_id]["P&L MONEY"] = value - pos["entry_value"]
        records.append({"Fecha": date, "Patrimonio": cash + invested})

    ledger_df = pd.DataFrame.from_dict(ledger, orient="index")
    ledger_df.index.name = "TRADE ID"
    return pd.DataFrame(records).set_index("Fecha"), ledger_df


def build_multiple_hedges(long_equity, hedges, prices, multiplier, today):
    total_pnl = pd.Series(0.0, index=long_equity.index)
    rows = []
    qqq = prices[HEDGE_TICKER].reindex(long_equity.index).ffill()

    for _, hedge in hedges.iterrows():
        open_session = next_session(hedge["OPEN DATE"], long_equity.index)
        close_session = pd.NaT if pd.isna(hedge["CLOSE DATE"]) else next_session(hedge["CLOSE DATE"], long_equity.index)
        if pd.isna(open_session):
            continue
        base_value = float(long_equity.loc[open_session])
        entry_price = float(qqq.loc[open_session])
        notional = base_value * float(multiplier)
        shares_short = notional / entry_price
        effective_end = long_equity.index[-1] if pd.isna(close_session) else min(close_session, long_equity.index[-1])
        exit_price = float(qqq.loc[effective_end])
        final_pnl = (entry_price - exit_price) * shares_short

        contribution = pd.Series(0.0, index=long_equity.index)
        active_mask = (contribution.index >= open_session) & (contribution.index <= effective_end)
        contribution.loc[active_mask] = (entry_price - qqq.loc[active_mask]) * shares_short
        if pd.notna(close_session):
            contribution.loc[contribution.index > effective_end] = final_pnl
        total_pnl += contribution

        rows.append({
            "Ticker": HEDGE_TICKER,
            "Operación": f"Cobertura corta ×{multiplier:g}",
            "Fecha apertura": open_session,
            "Fecha cierre": close_session,
            "Estado": "Abierta" if pd.isna(close_session) or close_session > today else "Cerrada",
            "Valor cartera apertura": base_value,
            "Notional inicial": notional,
            "Precio entrada QQQ": entry_price,
            "Precio final/actual QQQ": exit_price,
            "Ganancia / pérdida": final_pnl,
            "Rentabilidad": final_pnl / notional if notional else np.nan,
        })

    hedged_equity = long_equity + total_pnl
    return hedged_equity.rename("Cartera + coberturas"), total_pnl, pd.DataFrame(rows)


def buy_hold(prices, ticker, start_date, capital):
    series = prices[ticker].dropna().loc[start_date:]
    return series * (capital / series.iloc[0])


def select_dates(label, today):
    if label in MONTHS:
        month = MONTHS[label]
        start = pd.Timestamp(today.year, month, 1)
        return start, min(start + pd.offsets.MonthEnd(1), today), MONTH_NAMES[month]
    if label == "YTD":
        return pd.Timestamp(today.year, 1, 1), today, label
    return (today - QUICK_PERIODS[label]).normalize(), today, label


def stats(series):
    norm = series / series.iloc[0]
    dd = norm / norm.cummax() - 1
    ret = norm.iloc[-1] - 1
    max_dd = dd.min()
    return ret, max_dd, ret / abs(max_dd) if max_dd < 0 else np.nan


def money(v, signed=False):
    if pd.isna(v): return "-"
    return f"{'+' if signed and v > 0 else ''}{CURRENCY}{v:,.0f}"


def pct(v):
    return "N/D" if pd.isna(v) else f"{v:+.2%}"


def ratio(v):
    return "N/D" if pd.isna(v) or np.isinf(v) else f"{v:.2f}x"


def layout(fig, height, title, ytitle, money_axis=False):
    fig.update_layout(
        height=height, margin=dict(l=30, r=15, t=85, b=35),
        title=dict(text=title, x=.01, font=dict(color="#172033", size=18)),
        paper_bgcolor="#fff", plot_bgcolor="#f8fafc", font=dict(color="#334155", size=13),
        xaxis=dict(showgrid=False, linecolor="#94a3b8", automargin=True),
        yaxis=dict(title=ytitle, gridcolor="#dbe3eb", automargin=True, tickprefix=CURRENCY if money_axis else "", tickformat=",.0f" if money_axis else None, ticksuffix="" if money_axis else "%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )


def choose_quick():
    st.session_state.period_mode = "quick"
    st.session_state.month = None


def choose_month():
    st.session_state.period_mode = "month"
    st.session_state.quick = None


now = datetime.now(TZ)
today = pd.Timestamp(now.date())
st.session_state.setdefault("period_mode", "quick")
st.session_state.setdefault("quick", "YTD")
st.session_state.setdefault("month", None)

st.title("Rentabilidad de la cartera")
c1, c2, c3, c4 = st.columns([2.1, 1.8, 4.7, 1.4], vertical_alignment="bottom")
with c1:
    initial_capital = st.number_input("Capital inicial a 1 de enero", min_value=1_000.0, value=DEFAULT_CAPITAL, step=10_000.0, format="%.0f")
with c2:
    hedge_multiplier = st.number_input("Multiplicador coberturas", min_value=1.0, max_value=5.0, value=DEFAULT_HEDGE_MULTIPLIER, step=.1, format="%.1f")
with c3:
    st.radio("Periodo", ["1S", "1M", "3M", "6M", "YTD"], index=None, key="quick", horizontal=True, label_visibility="collapsed", on_change=choose_quick)
with c4:
    if st.button("↻ Actualizar", type="primary", width="stretch"):
        st.cache_data.clear(); st.rerun()
st.markdown('<div class="month-label">Seleccionar mes</div>', unsafe_allow_html=True)
st.radio("Mes", list(MONTHS.keys()), index=None, key="month", horizontal=True, label_visibility="collapsed", on_change=choose_month)

try:
    trades, hedges = load_workbook_data(FILE)
    tickers = tuple(sorted(set(trades["TICKER"]) | {SP500, HEDGE_TICKER}))
    first_date = min(trades["BUY DATE"].min(), hedges["OPEN DATE"].min())
    prices = get_prices(tickers, (first_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), (today + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
    if prices.empty: raise ValueError("No se han podido descargar precios.")

    missing = sorted(set(trades["TICKER"]) - set(prices.columns))
    if missing:
        st.warning("Sin precios para: " + ", ".join(missing))
        trades = trades[~trades["TICKER"].isin(missing)].copy()

    portfolio, ledger = build_long_portfolio(trades, prices, initial_capital)
    long_eq = portfolio["Patrimonio"].rename("Cartera")
    hedged_eq, hedge_pnl, hedge_history = build_multiple_hedges(long_eq, hedges, prices, hedge_multiplier, today)
    spy_eq = buy_hold(prices, SP500, long_eq.index[0], initial_capital).rename("S&P 500")
    all_eq = pd.concat([long_eq, hedged_eq, spy_eq], axis=1).ffill().dropna()

    label = st.session_state.month if st.session_state.period_mode == "month" and st.session_state.month else (st.session_state.quick or "YTD")
    start, end, period_label = select_dates(label, today)
    view = all_eq.loc[(all_eq.index >= start) & (all_eq.index <= end)].copy()
    if view.empty: raise ValueError("No hay sesiones para el periodo seleccionado.")

    long_ret, long_dd, long_ratio = stats(view["Cartera"])
    hedged_ret, hedged_dd, hedged_ratio = stats(view["Cartera + coberturas"])
    spy_ret, spy_dd, spy_ratio = stats(view["S&P 500"])

    # Grafica original de rentabilidad monetaria, sin cobertura.
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["Cartera"], name="Cartera", line=dict(color="#087f8c", width=3)))
    fig.add_trace(go.Scatter(x=view.index, y=view["S&P 500"], name="S&P 500", line=dict(color="#e07a3f", width=2.4)))
    layout(fig, 540, f"Valor de la cartera · {period_label}", "Valor", True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar":False, "displaylogo":False})

    st.markdown('<div class="section">Situación actual</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric(f"Cantidad inicial {period_label}", money(view["Cartera"].iloc[0]))
    b.metric(f"Cantidad final {period_label}", money(view["Cartera"].iloc[-1]))
    c.metric("Ganancia / pérdida", money(view["Cartera"].iloc[-1] - view["Cartera"].iloc[0], True))
    a, b, c = st.columns(3)
    a.metric(f"Cantidad inicial + hedge {period_label}", money(view["Cartera + coberturas"].iloc[0]))
    b.metric(f"Cantidad final + hedge {period_label}", money(view["Cartera + coberturas"].iloc[-1]))
    c.metric("Ganancia / pérdida + hedge", money(view["Cartera + coberturas"].iloc[-1] - view["Cartera + coberturas"].iloc[0], True))

    st.markdown('<div class="section">Cartera</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric(f"Rentabilidad {period_label}", pct(long_ret), delta=f"{long_ret-spy_ret:+.2%} vs S&P 500")
    b.metric("Drawdown máximo", pct(long_dd))
    c.metric("Rentabilidad / drawdown", ratio(long_ratio))
    a, b, c = st.columns(3)
    a.metric(f"Rentabilidad {period_label} hedge", pct(hedged_ret))
    b.metric("Drawdown máximo hedge", pct(hedged_dd))
    c.metric("Rentabilidad / drawdown hedge", ratio(hedged_ratio))

    st.markdown('<div class="section">S&P 500</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric(f"Rentabilidad {period_label}", pct(spy_ret))
    b.metric("Drawdown máximo", pct(spy_dd))
    c.metric("Rentabilidad / drawdown", ratio(spy_ratio))

    # Grafica original de drawdown, sin cobertura.
    dd = view / view.cummax() - 1
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=dd.index, y=dd["Cartera"]*100, name="Cartera", line=dict(color="#8b1e3f", width=2.6), fill="tozeroy", fillcolor="rgba(139,30,63,.14)"))
    dd_fig.add_trace(go.Scatter(x=dd.index, y=dd["S&P 500"]*100, name="S&P 500", line=dict(color="#e07a3f", width=2.2)))
    layout(dd_fig, 410, f"Curva de drawdown · {period_label}", "Drawdown")
    st.plotly_chart(dd_fig, width="stretch", config={"displayModeBar":False, "displaylogo":False})

    # Nueva grafica específica de cartera + coberturas.
    hedge_fig = go.Figure()
    hedge_fig.add_trace(go.Scatter(x=view.index, y=view["Cartera + coberturas"], name=f"Cartera + coberturas ×{hedge_multiplier:g}", line=dict(color="#6f2dbd", width=3)))
    hedge_fig.add_trace(go.Scatter(x=view.index, y=view["Cartera"], name="Cartera sin coberturas", line=dict(color="#087f8c", width=2, dash="dot")))
    hedge_fig.add_trace(go.Scatter(x=view.index, y=view["S&P 500"], name="S&P 500", line=dict(color="#e07a3f", width=2)))
    layout(hedge_fig, 470, f"Cartera + coberturas · multiplicador ×{hedge_multiplier:g}", "Valor", True)
    st.plotly_chart(hedge_fig, width="stretch", config={"displayModeBar":False, "displaylogo":False})

    # Histórico normal.
    history = trades.merge(ledger, how="left", left_on="TRADE ID", right_index=True)
    history["Estado"] = np.where(history["SELL DATE"].notna() & (history["SELL DATE"] <= today), "🔴 Cerrada", "🟢 Abierta")
    history["Final"] = history["SELL DATE"].where(history["SELL DATE"].notna(), today)
    history["Duración (días)"] = (history["Final"] - history["BUY DATE"]).dt.days
    history["Rentabilidad"] = history["P&L MONEY"] / history["ENTRY VALUE"]
    operations = history[["TICKER","COMPANY","GICS SECTOR","GICS SUB-INDUSTRY","BUY DATE","SELL DATE","Estado","Duración (días)","ENTRY VALUE","CURRENT VALUE","P&L MONEY","Rentabilidad"]].copy()
    operations.columns = ["Ticker","Compañía","Sector GICS","Subindustria GICS","Fecha compra","Fecha cierre","Estado","Duración (días)","Importe inicial","Importe final/actual","Ganancia / pérdida","Rentabilidad"]
    operations = operations.sort_values(["Estado","Fecha compra"], ascending=[False,False])
    operations["Fecha compra"] = operations["Fecha compra"].dt.strftime("%d/%m/%Y")
    operations["Fecha cierre"] = operations["Fecha cierre"].dt.strftime("%d/%m/%Y").fillna("-")

    def color(v):
        if pd.isna(v): return "color:#64748b;font-weight:700"
        return "color:#0a8f55;font-weight:800" if v > 0 else "color:#d93025;font-weight:800" if v < 0 else "font-weight:800"
    styled = operations.style.map(color, subset=["Ganancia / pérdida","Rentabilidad"]).format({
        "Importe inicial":lambda v:money(v), "Importe final/actual":lambda v:money(v),
        "Ganancia / pérdida":lambda v:money(v,True), "Rentabilidad":lambda v:pct(v),
        "Duración (días)":lambda v:f"{int(v)} días",
    })
    st.subheader(f"Histórico de operaciones {today.year}")
    st.dataframe(styled, width="stretch", hide_index=True)

    # Histórico específico de coberturas.
    hedge_display = hedge_history.copy()
    hedge_display["Fecha apertura"] = hedge_display["Fecha apertura"].dt.strftime("%d/%m/%Y")
    hedge_display["Fecha cierre"] = hedge_display["Fecha cierre"].dt.strftime("%d/%m/%Y").fillna("-")
    hedge_display["Estado"] = hedge_display["Estado"].map({"Abierta":"🟢 Abierta","Cerrada":"🔴 Cerrada"})
    hedge_styled = hedge_display.style.map(color, subset=["Ganancia / pérdida","Rentabilidad"]).format({
        "Valor cartera apertura":lambda v:money(v), "Notional inicial":lambda v:money(v),
        "Precio entrada QQQ":lambda v:f"{CURRENCY}{v:,.2f}", "Precio final/actual QQQ":lambda v:f"{CURRENCY}{v:,.2f}",
        "Ganancia / pérdida":lambda v:money(v,True), "Rentabilidad":lambda v:pct(v),
    })
    st.subheader("Histórico de operaciones hedge")
    st.caption(f"Cada cobertura usa ×{hedge_multiplier:g} del valor de mercado de la cartera en su fecha de apertura.")
    st.dataframe(hedge_styled, width="stretch", hide_index=True)

    st.caption("Las coberturas son posiciones cortas en QQQ. SELL DATE abre el corto y BUY DATE lo cierra. No se incluyen comisiones, financiación, dividendos debitados, slippage ni impuestos.")

except Exception as exc:
    st.error(f"No se ha podido generar el dashboard: {exc}")
    st.exception(exc)
