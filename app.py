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
@media(max-width:768px){.block-container{padding:1rem .75rem 2rem!important}h1{font-size:2rem!important;line-height:1.15!important}div[role="radiogroup"]{gap:.25rem!important}div[role="radiogroup"] label{min-width:48px!important;padding:.28rem .42rem!important}div[role="radiogroup"] label p{font-size:.82rem!important}}
</style>
""",
    unsafe_allow_html=True,
)


def load_workbook_data(path):
    raw = pd.read_excel(path, header=None, engine="openpyxl")
    header_rows = raw.index[
        raw.iloc[:, 0].astype(str).str.strip().str.upper().eq("TICKER")
    ].tolist()
    if not header_rows:
        raise ValueError("No encuentro la cabecera principal del Excel.")

    first_header = header_rows[0]
    second_header = header_rows[1] if len(header_rows) > 1 else len(raw)

    trades = raw.iloc[first_header + 1:second_header, :6].copy()
    trades.columns = [
        "TICKER", "COMPANY", "GICS SECTOR", "GICS SUB-INDUSTRY",
        "BUY DATE", "SELL DATE",
    ]
    trades = trades[trades["TICKER"].notna()].copy()
    trades["TICKER"] = (
        trades["TICKER"].astype(str).str.strip().str.upper().replace({"SATS": "ECHO"})
    )
    trades["BUY DATE"] = pd.to_datetime(trades["BUY DATE"], errors="coerce").dt.normalize()
    trades["SELL DATE"] = pd.to_datetime(trades["SELL DATE"], errors="coerce").dt.normalize()
    trades = trades[trades["BUY DATE"].notna()].reset_index(drop=True)
    trades["TRADE ID"] = np.arange(len(trades))

    hedges = pd.DataFrame(columns=["TICKER", "COMPANY", "OPEN DATE", "CLOSE DATE"])
    if len(header_rows) > 1:
        hedges = raw.iloc[second_header + 1:, :4].copy()
        # En la tabla hedge: SELL DATE abre el corto y BUY DATE lo cierra.
        hedges.columns = ["TICKER", "COMPANY", "OPEN DATE", "CLOSE DATE"]
        hedges = hedges[hedges["TICKER"].notna()].copy()
        hedges["TICKER"] = hedges["TICKER"].astype(str).str.strip().str.upper()
        hedges["OPEN DATE"] = pd.to_datetime(hedges["OPEN DATE"], errors="coerce").dt.normalize()
        hedges["CLOSE DATE"] = pd.to_datetime(hedges["CLOSE DATE"], errors="coerce").dt.normalize()
        hedges = hedges[
            hedges["TICKER"].eq(HEDGE_TICKER) & hedges["OPEN DATE"].notna()
        ].reset_index(drop=True)

    if trades.empty:
        raise ValueError("No hay operaciones de cartera válidas.")
    if hedges.empty:
        raise ValueError("No hay operaciones de cobertura QQQ válidas.")
    return trades, hedges


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
    event_dates = sorted(
        set(events["BUY SESSION"].dropna()) | set(events["SELL SESSION"].dropna())
    )
    if not event_dates:
        raise ValueError("No hay fechas de operación válidas.")

    first_event = event_dates[0]
    holdings = {}
    ledger = {}
    records = []
    cash = float(initial_capital)

    for date in sessions[sessions >= first_event]:
        prices_today = prices.loc[date]

        if date in event_dates:
            sales = events[events["SELL SESSION"].eq(date)]
            purchases = events[events["BUY SESSION"].eq(date)]
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
                allocation = investment_pool / len(valid_buys)
                for trade in valid_buys:
                    trade_id = int(trade["TRADE ID"])
                    ticker = trade["TICKER"]
                    shares = allocation / prices_today[ticker]
                    holdings[trade_id] = {
                        "ticker": ticker,
                        "shares": shares,
                        "entry_value": allocation,
                    }
                    cash -= allocation
                    ledger[trade_id] = {
                        "ENTRY VALUE": allocation,
                        "CURRENT VALUE": allocation,
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

    ledger_df = pd.DataFrame.from_dict(ledger, orient="index")
    ledger_df.index.name = "TRADE ID"
    return pd.DataFrame(records).set_index("Fecha"), ledger_df


def build_multiple_hedges(long_equity, hedges, prices, multiplier, today):
    total_pnl = pd.Series(0.0, index=long_equity.index)
    history_rows = []
    qqq = prices[HEDGE_TICKER].reindex(long_equity.index).ffill()

    for _, hedge in hedges.iterrows():
        open_session = next_session(hedge["OPEN DATE"], long_equity.index)
        close_session = (
            pd.NaT if pd.isna(hedge["CLOSE DATE"])
            else next_session(hedge["CLOSE DATE"], long_equity.index)
        )
        if pd.isna(open_session):
            continue

        base_value = float(long_equity.loc[open_session])
        entry_price = float(qqq.loc[open_session])
        notional = base_value * float(multiplier)
        short_shares = notional / entry_price
        effective_end = (
            long_equity.index[-1]
            if pd.isna(close_session)
            else min(close_session, long_equity.index[-1])
        )
        exit_price = float(qqq.loc[effective_end])
        final_pnl = (entry_price - exit_price) * short_shares

        contribution = pd.Series(0.0, index=long_equity.index)
        active = (contribution.index >= open_session) & (contribution.index <= effective_end)
        contribution.loc[active] = (
            entry_price - qqq.loc[active]
        ) * short_shares
        if pd.notna(close_session):
            contribution.loc[contribution.index > effective_end] = final_pnl
        total_pnl += contribution

        history_rows.append({
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
    return hedged_equity.rename("Cartera + coberturas"), pd.DataFrame(history_rows)


def buy_hold(prices, ticker, start_date, capital):
    series = prices[ticker].dropna().loc[start_date:]
    return series * (capital / series.iloc[0])


def selected_range(label, today):
    if label in MONTHS:
        month = MONTHS[label]
        start = pd.Timestamp(today.year, month, 1)
        return start, min(start + pd.offsets.MonthEnd(1), today), MONTH_NAMES[month]
    if label == "YTD":
        return pd.Timestamp(today.year, 1, 1), today, "YTD"
    return (today - QUICK_PERIODS[label]).normalize(), today, label


def calculate_stats(series):
    normalized = series / series.iloc[0]
    drawdown = normalized / normalized.cummax() - 1
    total_return = normalized.iloc[-1] - 1
    max_drawdown = drawdown.min()
    ratio_value = total_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    return total_return, max_drawdown, ratio_value


def money(value, signed=False):
    if pd.isna(value):
        return "-"
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{CURRENCY}{value:,.0f}"


def pct(value):
    return "N/D" if pd.isna(value) else f"{value:+.2%}"


def ratio(value):
    return "N/D" if pd.isna(value) or np.isinf(value) else f"{value:.2f}x"


def apply_layout(fig, height, title, y_title, money_axis=False):
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=15, t=85, b=35),
        title=dict(text=title, x=.01, font=dict(color="#172033", size=18)),
        paper_bgcolor="#fff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#334155", size=13),
        xaxis=dict(showgrid=False, linecolor="#94a3b8", automargin=True),
        yaxis=dict(
            title=y_title,
            gridcolor="#dbe3eb",
            automargin=True,
            tickprefix=CURRENCY if money_axis else "",
            tickformat=",.0f" if money_axis else None,
            ticksuffix="" if money_axis else "%",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        hovermode="x unified",
    )


def select_quick_period():
    st.session_state.period_mode = "quick"
    st.session_state.month_choice = None


def select_month():
    st.session_state.period_mode = "month"
    st.session_state.quick_choice = None


now = datetime.now(TZ)
today = pd.Timestamp(now.date())
st.session_state.setdefault("period_mode", "quick")
st.session_state.setdefault("quick_choice", "YTD")
st.session_state.setdefault("month_choice", None)

st.title("Rentabilidad de la cartera")
capital_col, hedge_col, period_col, refresh_col = st.columns(
    [2.1, 1.8, 4.7, 1.4], vertical_alignment="bottom"
)
with capital_col:
    initial_capital = st.number_input(
        "Capital inicial a 1 de enero",
        min_value=1_000.0,
        value=DEFAULT_CAPITAL,
        step=10_000.0,
        format="%.0f",
    )
with hedge_col:
    hedge_multiplier = st.number_input(
        "Multiplicador coberturas",
        min_value=1.0,
        max_value=5.0,
        value=DEFAULT_HEDGE_MULTIPLIER,
        step=.1,
        format="%.1f",
    )
with period_col:
    st.radio(
        "Periodo", ["1S", "1M", "3M", "6M", "YTD"],
        index=None, key="quick_choice", horizontal=True,
        label_visibility="collapsed", on_change=select_quick_period,
    )
with refresh_col:
    if st.button("↻ Actualizar", type="primary", width="stretch"):
        st.cache_data.clear()
        st.rerun()

st.markdown('<div class="month-label">Seleccionar mes</div>', unsafe_allow_html=True)
st.radio(
    "Mes", list(MONTHS.keys()), index=None, key="month_choice",
    horizontal=True, label_visibility="collapsed", on_change=select_month,
)

try:
    trades, hedges = load_workbook_data(FILE)
    tickers = tuple(sorted(set(trades["TICKER"]) | {SP500, NASDAQ100, HEDGE_TICKER}))
    first_date = min(trades["BUY DATE"].min(), hedges["OPEN DATE"].min())
    prices = get_prices(
        tickers,
        (first_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        (today + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
    )
    if prices.empty:
        raise ValueError("No se han podido descargar precios.")

    missing = sorted(set(trades["TICKER"]) - set(prices.columns))
    if missing:
        st.warning("Sin precios para: " + ", ".join(missing))
        trades = trades[~trades["TICKER"].isin(missing)].copy()

    portfolio, ledger = build_long_portfolio(trades, prices, initial_capital)
    long_equity = portfolio["Patrimonio"].rename("Cartera")
    hedged_equity, hedge_history = build_multiple_hedges(
        long_equity, hedges, prices, hedge_multiplier, today
    )
    spy_equity = buy_hold(
        prices, SP500, long_equity.index[0], initial_capital
    ).rename("S&P 500")
    nasdaq_equity = buy_hold(
        prices, NASDAQ100, long_equity.index[0], initial_capital
    ).rename("Nasdaq 100")
    all_equity = pd.concat(
        [long_equity, hedged_equity, spy_equity, nasdaq_equity], axis=1
    ).ffill().dropna()

    selector = (
        st.session_state.month_choice
        if st.session_state.period_mode == "month" and st.session_state.month_choice
        else (st.session_state.quick_choice or "YTD")
    )
    start, end, period_label = selected_range(selector, today)
    view = all_equity.loc[
        (all_equity.index >= start) & (all_equity.index <= end)
    ].copy()
    if view.empty:
        raise ValueError("No hay sesiones para el periodo seleccionado.")

    long_return, long_drawdown, long_ratio = calculate_stats(view["Cartera"])
    hedge_return, hedge_drawdown, hedge_ratio = calculate_stats(
        view["Cartera + coberturas"]
    )
    spy_return, spy_drawdown, spy_ratio = calculate_stats(view["S&P 500"])
    nasdaq_return, nasdaq_drawdown, nasdaq_ratio = calculate_stats(view["Nasdaq 100"])

    # GRÁFICO PRINCIPAL: ahora incluye cartera + coberturas.
    performance_fig = go.Figure()
    performance_fig.add_trace(go.Scatter(
        x=view.index, y=view["Cartera"],
        name="Cartera sin coberturas",
        line=dict(color="#087f8c", width=3),
        hovertemplate=f"%{{x|%d/%m/%Y}}<br><b>{CURRENCY}%{{y:,.0f}}</b><extra>Sin coberturas</extra>",
    ))
    performance_fig.add_trace(go.Scatter(
        x=view.index, y=view["Cartera + coberturas"],
        name=f"Cartera + coberturas ×{hedge_multiplier:g}",
        line=dict(color="#6f2dbd", width=3),
        hovertemplate=f"%{{x|%d/%m/%Y}}<br><b>{CURRENCY}%{{y:,.0f}}</b><extra>Con coberturas</extra>",
    ))
    performance_fig.add_trace(go.Scatter(
        x=view.index, y=view["S&P 500"],
        name="S&P 500",
        line=dict(color="#e07a3f", width=2.4),
        hovertemplate=f"%{{x|%d/%m/%Y}}<br><b>{CURRENCY}%{{y:,.0f}}</b><extra>S&P 500</extra>",
    ))
    performance_fig.add_trace(go.Scatter(
        x=view.index, y=view["Nasdaq 100"],
        name="Nasdaq 100 (QQQ)",
        line=dict(color="#2563eb", width=2.4),
        hovertemplate=f"%{{x|%d/%m/%Y}}<br><b>{CURRENCY}%{{y:,.0f}}</b><extra>Nasdaq 100</extra>",
    ))
    apply_layout(
        performance_fig, 560,
        f"Valor de la cartera · {period_label}",
        "Valor", money_axis=True,
    )
    st.plotly_chart(
        performance_fig, width="stretch",
        config={"displayModeBar":False, "displaylogo":False, "responsive":True},
    )

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
    a.metric(f"Rentabilidad {period_label}",pct(long_return))
    b.metric("Drawdown máximo", pct(long_drawdown))
    c.metric("Rentabilidad / drawdown", ratio(long_ratio))
    a, b, c = st.columns(3)
    a.metric(f"Rentabilidad {period_label} hedge", pct(hedge_return))
    b.metric("Drawdown máximo hedge", pct(hedge_drawdown))
    c.metric("Rentabilidad / drawdown hedge", ratio(hedge_ratio))

    st.markdown('<div class="section">S&P 500</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric(f"Rentabilidad {period_label}", pct(spy_return))
    b.metric("Drawdown máximo", pct(spy_drawdown))
    c.metric("Rentabilidad / drawdown", ratio(spy_ratio))

    st.markdown('<div class="section">Nasdaq 100</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric(f"Rentabilidad {period_label}", pct(nasdaq_return))
    b.metric("Drawdown máximo", pct(nasdaq_drawdown))
    c.metric("Rentabilidad / drawdown", ratio(nasdaq_ratio))

    # DRAWDOWN: ahora incluye explícitamente cartera + coberturas.
    drawdown = view / view.cummax() - 1
    drawdown_fig = go.Figure()
    drawdown_fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown["Cartera"] * 100,
        name="Cartera sin coberturas",
        line=dict(color="#087f8c", width=2.6),
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>Sin coberturas</extra>",
    ))
    drawdown_fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown["Cartera + coberturas"] * 100,
        name=f"Cartera + coberturas ×{hedge_multiplier:g}",
        line=dict(color="#6f2dbd", width=2.8),
        fill="tozeroy",
        fillcolor="rgba(111,45,189,.12)",
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>Con coberturas</extra>",
    ))
    drawdown_fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown["S&P 500"] * 100,
        name="S&P 500",
        line=dict(color="#e07a3f", width=2.2),
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>S&P 500</extra>",
    ))
    drawdown_fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown["Nasdaq 100"] * 100,
        name="Nasdaq 100 (QQQ)",
        line=dict(color="#2563eb", width=2.2),
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>Nasdaq 100</extra>",
    ))
    drawdown_fig.add_hline(y=0, line_width=1, line_color="#94a3b8")
    apply_layout(
        drawdown_fig, 430,
        f"Curva de drawdown · {period_label}",
        "Drawdown",
    )
    st.plotly_chart(
        drawdown_fig, width="stretch",
        config={"displayModeBar":False, "displaylogo":False, "responsive":True},
    )

    # Histórico de operaciones de cartera.
    history = trades.merge(ledger, how="left", left_on="TRADE ID", right_index=True)
    history["Estado"] = np.where(
        history["SELL DATE"].notna() & (history["SELL DATE"] <= today),
        "🔴 Cerrada", "🟢 Abierta",
    )
    history["Final"] = history["SELL DATE"].where(history["SELL DATE"].notna(), today)
    history["Duración (días)"] = (history["Final"] - history["BUY DATE"]).dt.days
    history["Rentabilidad"] = history["P&L MONEY"] / history["ENTRY VALUE"]
    operations = history[[
        "TICKER", "COMPANY", "GICS SECTOR", "GICS SUB-INDUSTRY",
        "BUY DATE", "SELL DATE", "Estado", "Duración (días)",
        "ENTRY VALUE", "CURRENT VALUE", "P&L MONEY", "Rentabilidad",
    ]].copy()
    operations.columns = [
        "Ticker", "Compañía", "Sector GICS", "Subindustria GICS",
        "Fecha compra", "Fecha cierre", "Estado", "Duración (días)",
        "Importe inicial", "Importe final/actual", "Ganancia / pérdida", "Rentabilidad",
    ]
    operations = operations.sort_values(
        ["Estado", "Fecha compra"], ascending=[False, False]
    )
    operations["Fecha compra"] = operations["Fecha compra"].dt.strftime("%d/%m/%Y")
    operations["Fecha cierre"] = operations["Fecha cierre"].dt.strftime("%d/%m/%Y").fillna("-")

    def color_result(value):
        if pd.isna(value):
            return "color:#64748b;font-weight:700"
        if value > 0:
            return "color:#0a8f55;font-weight:800"
        if value < 0:
            return "color:#d93025;font-weight:800"
        return "font-weight:800"

    operations_styled = (
        operations.style
        .map(color_result, subset=["Ganancia / pérdida", "Rentabilidad"])
        .format({
            "Importe inicial": lambda value: money(value),
            "Importe final/actual": lambda value: money(value),
            "Ganancia / pérdida": lambda value: money(value, True),
            "Rentabilidad": lambda value: pct(value),
            "Duración (días)": lambda value: f"{int(value)} días",
        })
    )
    st.subheader(f"Histórico de operaciones {today.year}")
    st.dataframe(operations_styled, width="stretch", hide_index=True)

    # Histórico independiente de coberturas.
    hedge_display = hedge_history.copy()
    hedge_display["Fecha apertura"] = hedge_display["Fecha apertura"].dt.strftime("%d/%m/%Y")
    hedge_display["Fecha cierre"] = hedge_display["Fecha cierre"].dt.strftime("%d/%m/%Y").fillna("-")
    hedge_display["Estado"] = hedge_display["Estado"].map({
        "Abierta": "🟢 Abierta", "Cerrada": "🔴 Cerrada",
    })
    hedge_styled = (
        hedge_display.style
        .map(color_result, subset=["Ganancia / pérdida", "Rentabilidad"])
        .format({
            "Valor cartera apertura": lambda value: money(value),
            "Notional inicial": lambda value: money(value),
            "Precio entrada QQQ": lambda value: f"{CURRENCY}{value:,.2f}",
            "Precio final/actual QQQ": lambda value: f"{CURRENCY}{value:,.2f}",
            "Ganancia / pérdida": lambda value: money(value, True),
            "Rentabilidad": lambda value: pct(value),
        })
    )
    st.subheader("Histórico de operaciones hedge")
    st.caption(
        f"Cada cobertura utiliza ×{hedge_multiplier:g} del valor de mercado "
        "de la cartera en su fecha de apertura."
    )
    st.dataframe(hedge_styled, width="stretch", hide_index=True)

    st.caption(
        "Las coberturas son posiciones cortas en QQQ. En la segunda tabla del Excel, "
        "SELL DATE abre el corto y BUY DATE lo cierra. No se incluyen comisiones, "
        "financiación, dividendos debitados, slippage ni impuestos."
    )

except Exception as exc:
    st.error(f"No se ha podido generar el dashboard: {exc}")
    st.exception(exc)
