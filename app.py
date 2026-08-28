from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

FILE = Path(__file__).with_name("USAstocks.xlsx")
BENCHMARK = "SPY"
DEFAULT_INITIAL_CAPITAL = 100_000.0
CURRENCY = "$"
TZ = ZoneInfo("Europe/Madrid")
PERIODS = {
    "1S": pd.DateOffset(weeks=1),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
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
:root {color-scheme:light!important}
html,body,[data-testid="stAppViewContainer"],.stApp{background:#fff!important;color:#172033!important}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important}
.block-container{padding-top:1.2rem;padding-bottom:2.5rem;max-width:1450px}
h1{font-size:2rem!important;color:#132238!important;letter-spacing:-.03em}
[data-testid="stMetric"]{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:14px 16px;box-shadow:0 4px 14px rgba(15,23,42,.055)}
[data-testid="stMetricLabel"] p{color:#64748b!important}
[data-testid="stMetricValue"]{color:#172033!important;font-weight:750}
div[role="radiogroup"]{gap:.35rem;flex-wrap:wrap!important}
div[role="radiogroup"] label{background:#f1f5f9!important;border:1px solid #cbd5e1!important;border-radius:10px!important;padding:.35rem .8rem!important;min-width:55px!important;justify-content:center!important;color:#172033!important}
div[role="radiogroup"] label p,div[role="radiogroup"] label span{color:#172033!important;font-weight:700!important;opacity:1!important}
div[role="radiogroup"] label:has(input:checked){background:#143d59!important;border-color:#143d59!important}
div[role="radiogroup"] label:has(input:checked) p,div[role="radiogroup"] label:has(input:checked) span{color:#fff!important;font-weight:800!important}
.section{font-size:.82rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:750;margin:.7rem 0 .55rem}
@media(max-width:768px){.block-container{padding:1rem .75rem 2rem!important}h1{font-size:2rem!important;line-height:1.15!important}div[role="radiogroup"]{gap:.3rem!important}div[role="radiogroup"] label{min-width:58px!important;padding:.3rem .55rem!important}div[role="radiogroup"] label p{font-size:.9rem!important}}
</style>
""",
    unsafe_allow_html=True,
)


def load_trades(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(column).strip().upper() for column in df.columns]
    missing = {"TICKER", "BUY DATE", "SELL DATE"} - set(df.columns)
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(sorted(missing)))
    df["TICKER"] = df["TICKER"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    df["TICKER"] = df["TICKER"].replace({"SATS": "ECHO"})
    df["BUY DATE"] = pd.to_datetime(df["BUY DATE"], errors="coerce").dt.normalize()
    df["SELL DATE"] = pd.to_datetime(df["SELL DATE"], errors="coerce").dt.normalize()
    df = df[df["BUY DATE"].notna() & df["TICKER"].ne("")].copy()
    if df.empty:
        raise ValueError("El Excel no contiene operaciones válidas.")
    if (df["SELL DATE"].notna() & (df["SELL DATE"] < df["BUY DATE"])).any():
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
    prices = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    return prices.sort_index().dropna(how="all")


def next_session(date, sessions):
    candidates = sessions[sessions >= pd.Timestamp(date)]
    return candidates[0] if len(candidates) else pd.NaT


def prepare_events(trades, sessions):
    events = trades.copy()
    events["BUY SESSION"] = events["BUY DATE"].map(lambda value: next_session(value, sessions))
    events["SELL SESSION"] = events["SELL DATE"].map(
        lambda value: pd.NaT if pd.isna(value) else next_session(value, sessions)
    )
    return events


def build_money_portfolio(trades, prices, initial_capital):
    """Cartera monetaria sin rebalanceo. Las ventas financian las compras de la misma sesión."""
    sessions = prices.index
    events = prepare_events(trades, sessions)
    event_dates = sorted(set(events["BUY SESSION"].dropna()) | set(events["SELL SESSION"].dropna()))
    if not event_dates:
        raise ValueError("No se han encontrado fechas de operación válidas.")

    first_event = event_dates[0]
    holdings = {}  # trade_id -> datos de la posición
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

        records.append({
            "Fecha": date,
            "Patrimonio": cash + invested,
            "Invertido": invested,
            "Efectivo": cash,
        })

    portfolio = pd.DataFrame(records).set_index("Fecha")
    ledger_df = pd.DataFrame.from_dict(ledger, orient="index")
    ledger_df.index.name = "TRADE ID"
    return portfolio, events, ledger_df


def build_benchmark(prices, start_date, initial_capital):
    benchmark = prices[BENCHMARK].dropna().loc[start_date:]
    shares = initial_capital / benchmark.iloc[0]
    return benchmark * shares


def period_start(period, today):
    return pd.Timestamp(today.year, 1, 1) if period == "YTD" else (today - PERIODS[period]).normalize()


def period_values(values, start):
    selected = values.loc[values.index >= start].copy()
    if selected.empty:
        raise ValueError("No hay sesiones en el periodo seleccionado.")
    return selected


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


def pct(value):
    return "N/D" if pd.isna(value) else f"{value:+.2%}"


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


now = datetime.now(TZ)
today = pd.Timestamp(now.date())
st.title("Rentabilidad de la cartera")

control_money, control_period, refresh_col = st.columns([2.2, 5.8, 1.5], vertical_alignment="bottom")
with control_money:
    initial_capital = st.number_input(
        "Capital inicial a 1 de enero",
        min_value=1_000.0,
        value=DEFAULT_INITIAL_CAPITAL,
        step=10_000.0,
        format="%.0f",
    )
with control_period:
    period = st.radio(
        "Periodo", ["1S", "1M", "3M", "6M", "YTD"],
        index=4, horizontal=True, label_visibility="collapsed",
    )
with refresh_col:
    if st.button("↻ Actualizar", type="primary", width="stretch"):
        st.cache_data.clear()
        st.rerun()

if not FILE.exists():
    st.error("No encuentro USAstocks.xlsx junto a app.py.")
    st.stop()

try:
    trades = load_trades(FILE)
    tickers = tuple(sorted(set(trades["TICKER"]) | {BENCHMARK}))
    prices = get_prices(
        tickers,
        (trades["BUY DATE"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        (today + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
    )
    if prices.empty or BENCHMARK not in prices.columns:
        st.error("No se han podido descargar los precios.")
        st.stop()

    missing = sorted(set(trades["TICKER"]) - set(prices.columns))
    if missing:
        st.warning("Sin precios para: " + ", ".join(missing))
        trades = trades[~trades["TICKER"].isin(missing)].copy()

    portfolio, prepared_trades, ledger = build_money_portfolio(trades, prices, initial_capital)
    benchmark = build_benchmark(prices, portfolio.index[0], initial_capital)
    equity = pd.concat([
        portfolio["Patrimonio"].rename("Cartera"),
        benchmark.rename("S&P 500"),
    ], axis=1).ffill().dropna()

    selected = period_values(equity, period_start(period, today))
    portfolio_stats = calculate_stats(selected["Cartera"])
    benchmark_stats = calculate_stats(selected["S&P 500"])
    drawdown = selected / selected.cummax() - 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=selected.index, y=selected["Cartera"], name="Cartera", mode="lines",
        line=dict(color="#087f8c", width=3),
        hovertemplate=f"%{{x|%d/%m/%Y}}<br><b>{CURRENCY}%{{y:,.0f}}</b><extra>Cartera</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=selected.index, y=selected["S&P 500"], name="S&P 500 (SPY)", mode="lines",
        line=dict(color="#e07a3f", width=2.6),
        hovertemplate=f"%{{x|%d/%m/%Y}}<br><b>{CURRENCY}%{{y:,.0f}}</b><extra>S&P 500</extra>",
    ))
    common_layout(fig, 555, f"Valor de la cartera {period} · Última sesión: {selected.index[-1]:%d/%m/%Y}")
    fig.update_yaxes(
        title="Valor de la cartera", tickprefix=CURRENCY, tickformat=",.0f",
        gridcolor="#dbe3eb", tickfont=dict(color="#334155", size=12), automargin=True,
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo":False, "displayModeBar":False, "responsive":True})

    st.markdown('<div class="section">Cartera</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Rentabilidad {period}", pct(portfolio_stats["return"]), delta=f'{portfolio_stats["return"] - benchmark_stats["return"]:+.2%} vs S&P 500')
    c2.metric("Drawdown máximo", pct(portfolio_stats["dd"]))
    c3.metric("Rentabilidad / drawdown", ratio(portfolio_stats["ratio"]))

    st.markdown('<div class="section">S&P 500</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.metric(f"Rentabilidad {period}", pct(benchmark_stats["return"]))
    s2.metric("Drawdown máximo", pct(benchmark_stats["dd"]))
    s3.metric("Rentabilidad / drawdown", ratio(benchmark_stats["ratio"]))

    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown["Cartera"] * 100,
        name="Drawdown cartera", mode="lines",
        line=dict(color="#8b1e3f", width=2.6), fill="tozeroy",
        fillcolor="rgba(139,30,63,.16)",
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>Cartera</extra>",
    ))
    dd_fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown["S&P 500"] * 100,
        name="Drawdown S&P 500", mode="lines",
        line=dict(color="#e07a3f", width=2.2),
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>S&P 500</extra>",
    ))
    dd_fig.add_hline(y=0, line_width=1, line_color="#94a3b8")
    common_layout(dd_fig, 420, f"Curva de drawdown {period}")
    dd_fig.update_yaxes(title="Drawdown", ticksuffix="%", gridcolor="#dbe3eb", rangemode="tozero")
    st.plotly_chart(dd_fig, width="stretch", config={"displaylogo":False, "displayModeBar":False, "responsive":True})

    initial_period_value = selected["Cartera"].iloc[0]
    current_value = selected["Cartera"].iloc[-1]
    period_result = current_value - initial_period_value
    st.markdown('<div class="section">Situación actual</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric(f"Cantidad inicial {period}", money(initial_period_value))
    p2.metric("Cantidad actual", money(current_value))
    p3.metric(f"Ganancia / pérdida {period}", money(period_result, signed=True))

    year_start = pd.Timestamp(today.year, 1, 1)
    history = trades[(trades["BUY DATE"] >= year_start) & (trades["BUY DATE"] <= today)].copy()
    history = history.merge(ledger, how="left", left_on="TRADE ID", right_index=True)
    history["ESTADO"] = np.where(history["SELL DATE"].notna() & (history["SELL DATE"] <= today), "Cerrada", "Abierta")
    history["FINAL"] = history["SELL DATE"].where(history["ESTADO"].eq("Cerrada"), today)
    history["DURACION"] = (history["FINAL"] - history["BUY DATE"]).dt.days
    history["RENTABILIDAD"] = history["P&L MONEY"] / history["ENTRY VALUE"]
    open_count = int(history["ESTADO"].eq("Abierta").sum())
    closed_count = int(history["ESTADO"].eq("Cerrada").sum())

    st.subheader(f"Histórico de operaciones {today.year}")
    st.caption(f"{len(history)} operaciones desde el 1 de enero · {open_count} abiertas · {closed_count} cerradas")
    wanted = [
        "TICKER", "COMPANY", "GICS SECTOR", "GICS INDUSTRY", "GICS SUB-INDUSTRY",
        "BUY DATE", "SELL DATE", "ESTADO", "DURACION", "ENTRY VALUE",
        "CURRENT VALUE", "P&L MONEY", "RENTABILIDAD",
    ]
    operations = history[[column for column in wanted if column in history.columns]].copy()
    operations["ORDEN ESTADO"] = operations["ESTADO"].map({"Abierta":0, "Cerrada":1})
    operations = operations.sort_values(["ORDEN ESTADO", "BUY DATE", "TICKER"], ascending=[True, False, True]).drop(columns="ORDEN ESTADO")
    operations["BUY DATE"] = operations["BUY DATE"].dt.strftime("%d/%m/%Y")
    operations["SELL DATE"] = operations["SELL DATE"].dt.strftime("%d/%m/%Y").fillna("-")
    operations["ESTADO"] = operations["ESTADO"].map({"Abierta":"🟢 Abierta", "Cerrada":"🔴 Cerrada"})
    operations = operations.rename(columns={
        "TICKER":"Ticker", "COMPANY":"Compañía", "GICS SECTOR":"Sector GICS",
        "GICS INDUSTRY":"Industria GICS", "GICS SUB-INDUSTRY":"Subindustria GICS",
        "BUY DATE":"Fecha de compra", "SELL DATE":"Fecha de cierre", "ESTADO":"Estado",
        "DURACION":"Duración (días)", "ENTRY VALUE":"Importe inicial",
        "CURRENT VALUE":"Importe final/actual", "P&L MONEY":"Ganancia / pérdida",
        "RENTABILIDAD":"Rentabilidad",
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
        "Simulación monetaria sin rebalanceo diario. En cada cambio mensual, el importe "
        "de las ventas se distribuye por igual entre las compras de la misma fecha. "
        "Se permiten acciones fraccionarias y no se incluyen comisiones ni impuestos."
    )

except Exception as exc:
    st.error(f"No se ha podido generar el dashboard: {exc}")
    st.exception(exc)
