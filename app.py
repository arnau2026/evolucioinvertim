from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =========================
# CONFIGURACION
# =========================
FILE = Path(__file__).with_name("USAstocks.xlsx")
BENCHMARK = "SPY"
INITIAL_CAPITAL = 100_000.0
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
:root {color-scheme: light !important;}
html, body, [data-testid="stAppViewContainer"], .stApp {
    background:#ffffff !important;
    color:#172033 !important;
}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none!important;}
.block-container {padding-top:1.2rem; padding-bottom:2.5rem; max-width:1450px;}
h1 {font-size:2rem!important; color:#132238!important; letter-spacing:-.03em;}
[data-testid="stMetric"] {
    background:#fff; border:1px solid #e2e8f0; border-radius:14px;
    padding:14px 16px; box-shadow:0 4px 14px rgba(15,23,42,.055);
}
[data-testid="stMetricLabel"] p {color:#64748b!important;}
[data-testid="stMetricValue"] {color:#172033!important; font-weight:750;}
div[role="radiogroup"] {gap:.35rem; flex-wrap:wrap!important;}
div[role="radiogroup"] label {
    background:#f1f5f9!important; border:1px solid #cbd5e1!important;
    border-radius:10px!important; padding:.35rem .8rem!important;
    min-width:55px!important; justify-content:center!important; color:#172033!important;
}
div[role="radiogroup"] label p,
div[role="radiogroup"] label [data-testid="stMarkdownContainer"],
div[role="radiogroup"] label span {
    color:#172033!important; font-weight:700!important; opacity:1!important;
}
div[role="radiogroup"] label:has(input:checked) {
    background:#143d59!important; border-color:#143d59!important;
}
div[role="radiogroup"] label:has(input:checked) p,
div[role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"],
div[role="radiogroup"] label:has(input:checked) span {
    color:#ffffff!important; font-weight:800!important; opacity:1!important;
}
.section {
    font-size:.82rem; color:#64748b; text-transform:uppercase;
    letter-spacing:.08em; font-weight:750; margin:.7rem 0 .55rem;
}
@media (max-width:768px) {
    .block-container {padding:1rem .75rem 2rem!important;}
    h1 {font-size:2rem!important; line-height:1.15!important;}
    div[role="radiogroup"] {gap:.3rem!important;}
    div[role="radiogroup"] label {min-width:58px!important; padding:.3rem .55rem!important;}
    div[role="radiogroup"] label p {font-size:.9rem!important;}
}
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

    df["TICKER"] = (
        df["TICKER"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    )
    df["TICKER"] = df["TICKER"].replace({"SATS": "ECHO"})
    df["BUY DATE"] = pd.to_datetime(df["BUY DATE"], errors="coerce").dt.normalize()
    df["SELL DATE"] = pd.to_datetime(df["SELL DATE"], errors="coerce").dt.normalize()
    df = df[df["BUY DATE"].notna() & df["TICKER"].ne("")].copy()

    if df.empty:
        raise ValueError("El Excel no contiene operaciones válidas.")
    invalid = df["SELL DATE"].notna() & (df["SELL DATE"] < df["BUY DATE"])
    if invalid.any():
        raise ValueError("Hay alguna fecha de venta anterior a la compra.")
    return df.sort_values(["BUY DATE", "TICKER"]).reset_index(drop=True)


@st.cache_data(ttl=55, show_spinner=False)
def get_prices(tickers, start, end):
    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        timeout=20,
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


def prepare_events(trades, sessions):
    """Convierte fechas del Excel en sesiones reales de mercado."""
    prepared = trades.copy()
    prepared["BUY SESSION"] = prepared["BUY DATE"].map(lambda value: next_session(value, sessions))
    prepared["SELL SESSION"] = prepared["SELL DATE"].map(
        lambda value: pd.NaT if pd.isna(value) else next_session(value, sessions)
    )
    return prepared


def build_money_portfolio(trades, prices, initial_capital):
    """
    Simulacion monetaria sin rebalanceo diario.

    - En la primera fecha de compra, todo el capital inicial se reparte por igual.
    - En fechas posteriores, primero se venden las posiciones señaladas.
    - Los ingresos de esas ventas se reparten por igual entre las compras de esa fecha.
    - Las posiciones que continúan abiertas conservan sus acciones y no se rebalancean.
    - Los posibles importes no invertidos permanecen en efectivo.
    - Se permiten acciones fraccionarias para repartir exactamente el importe disponible.
    """
    sessions = prices.index
    events = prepare_events(trades, sessions)
    event_dates = sorted(
        set(events["BUY SESSION"].dropna().tolist())
        | set(events["SELL SESSION"].dropna().tolist())
    )
    if not event_dates:
        raise ValueError("No se han encontrado fechas de operación válidas.")

    first_event = event_dates[0]
    holdings = {}  # ticker -> numero de acciones
    cash = float(initial_capital)
    records = []
    trade_log = []

    for date in sessions[sessions >= first_event]:
        prices_today = prices.loc[date]
        if date in event_dates:
            sales = events[events["SELL SESSION"] == date]
            purchases = events[events["BUY SESSION"] == date]

            sale_proceeds = 0.0
            for _, trade in sales.iterrows():
                ticker = trade["TICKER"]
                shares = holdings.pop(ticker, 0.0)
                price = prices_today.get(ticker, np.nan)
                if shares > 0 and pd.notna(price):
                    proceeds = shares * price
                    sale_proceeds += proceeds
                    cash += proceeds
                    trade_log.append({
                        "Fecha": date,
                        "Tipo": "Venta",
                        "Ticker": ticker,
                        "Importe": proceeds,
                    })

            valid_buys = []
            for _, trade in purchases.iterrows():
                ticker = trade["TICKER"]
                price = prices_today.get(ticker, np.nan)
                if pd.notna(price) and price > 0:
                    valid_buys.append(ticker)

            if valid_buys:
                if date == first_event:
                    investment_pool = cash
                else:
                    investment_pool = min(sale_proceeds, cash)

                amount_per_ticker = investment_pool / len(valid_buys)
                for ticker in valid_buys:
                    price = prices_today[ticker]
                    shares = amount_per_ticker / price
                    holdings[ticker] = holdings.get(ticker, 0.0) + shares
                    cash -= amount_per_ticker
                    trade_log.append({
                        "Fecha": date,
                        "Tipo": "Compra",
                        "Ticker": ticker,
                        "Importe": amount_per_ticker,
                    })

        invested = 0.0
        for ticker, shares in holdings.items():
            price = prices_today.get(ticker, np.nan)
            if pd.notna(price):
                invested += shares * price

        records.append({
            "Fecha": date,
            "Patrimonio": cash + invested,
            "Invertido": invested,
            "Efectivo": cash,
            "Posiciones": len(holdings),
        })

    portfolio = pd.DataFrame(records).set_index("Fecha")
    log = pd.DataFrame(trade_log)
    return portfolio, events, log


def build_benchmark(prices, start_date, initial_capital):
    benchmark = prices[BENCHMARK].dropna().loc[start_date:]
    if benchmark.empty:
        raise ValueError("No hay precios del S&P 500 para el periodo calculado.")
    shares = initial_capital / benchmark.iloc[0]
    return benchmark * shares


def period_start(period, today):
    if period == "YTD":
        return pd.Timestamp(today.year, 1, 1)
    return (today - PERIODS[period]).normalize()


def normalize_period(values, start):
    selected = values.loc[values.index >= start].copy()
    if selected.empty:
        raise ValueError("No hay sesiones en el periodo seleccionado.")
    return selected / selected.iloc[0] - 1


def calculate_stats(values, start):
    selected = values.loc[values.index >= start].copy()
    if selected.empty:
        return {"return": np.nan, "dd": np.nan, "ratio": np.nan}
    normalized = selected / selected.iloc[0]
    total_return = normalized.iloc[-1] - 1
    drawdown = normalized / normalized.cummax() - 1
    max_drawdown = drawdown.min()
    return {
        "return": total_return,
        "dd": max_drawdown,
        "ratio": total_return / abs(max_drawdown) if max_drawdown < 0 else np.nan,
    }


def operation_return(row, prices, today):
    ticker = row["TICKER"]
    if ticker not in prices.columns:
        return np.nan
    series = prices[ticker].dropna()
    buy_candidates = series.loc[series.index >= row["BUY DATE"]]
    if buy_candidates.empty:
        return np.nan
    end_date = today if pd.isna(row["SELL DATE"]) or row["SELL DATE"] > today else row["SELL DATE"]
    sell_candidates = series.loc[series.index <= end_date]
    if sell_candidates.empty or buy_candidates.iloc[0] <= 0:
        return np.nan
    return sell_candidates.iloc[-1] / buy_candidates.iloc[0] - 1


def pct(value):
    return "N/D" if pd.isna(value) else f"{value:+.2%}"


def ratio(value):
    return "N/D" if pd.isna(value) or np.isinf(value) else f"{value:.2f}x"


def money(value):
    return f"{CURRENCY}{value:,.0f}"


def common_layout(fig, height, title):
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=15, t=85, b=35),
        title=dict(text=title, x=0.01, font=dict(color="#172033", size=18)),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#334155", size=13),
        xaxis=dict(
            showgrid=False,
            linecolor="#94a3b8",
            tickfont=dict(color="#334155", size=12),
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#334155", size=12),
            bgcolor="rgba(255,255,255,0.85)",
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#111827",
            font=dict(color="#ffffff", size=13),
        ),
        hovermode="x unified",
    )


now = datetime.now(TZ)
today = pd.Timestamp(now.date())
st.title("Rentabilidad de la cartera")

if not FILE.exists():
    st.error("No encuentro USAstocks.xlsx junto a app.py.")
    st.stop()

try:
    trades = load_trades(FILE)
    download_start = (trades["BUY DATE"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    download_end = (today + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    tickers = tuple(sorted(set(trades["TICKER"]) | {BENCHMARK}))
    prices = get_prices(tickers, download_start, download_end)

    if prices.empty or BENCHMARK not in prices.columns:
        st.error("No se han podido descargar los precios.")
        st.stop()

    missing = sorted(set(trades["TICKER"]) - set(prices.columns))
    if missing:
        st.warning("Sin precios para: " + ", ".join(missing))
        trades = trades[~trades["TICKER"].isin(missing)].copy()

    portfolio, prepared_trades, transaction_log = build_money_portfolio(
        trades, prices, INITIAL_CAPITAL
    )
    benchmark = build_benchmark(prices, portfolio.index[0], INITIAL_CAPITAL)
    equity = pd.concat(
        [portfolio["Patrimonio"].rename("Cartera"), benchmark.rename("S&P 500")],
        axis=1,
    ).ffill().dropna()

    period_col, spacer_col, refresh_col = st.columns([5.0, 3.5, 1.5], vertical_alignment="center")
    with period_col:
        period = st.radio(
            "Periodo",
            ["1S", "1M", "3M", "6M", "YTD"],
            index=4,
            horizontal=True,
            label_visibility="collapsed",
        )
    with refresh_col:
        if st.button("↻ Actualizar", type="primary", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    start = period_start(period, today)
    normalized = normalize_period(equity, start)
    curve = normalized * 100
    drawdown = (1 + normalized) / (1 + normalized).cummax() - 1
    drawdown_pct = drawdown * 100
    portfolio_stats = calculate_stats(equity["Cartera"], start)
    benchmark_stats = calculate_stats(equity["S&P 500"], start)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve.index,
        y=curve["Cartera"],
        name="Cartera",
        mode="lines",
        line=dict(color="#087f8c", width=3),
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>Cartera</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=curve.index,
        y=curve["S&P 500"],
        name="S&P 500 (SPY)",
        mode="lines",
        line=dict(color="#e07a3f", width=2.6),
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>S&P 500</extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#9aa8b6")
    common_layout(
        fig,
        555,
        f"Rentabilidad acumulada {period} · Última sesión: {curve.index[-1]:%d/%m/%Y}",
    )
    fig.update_yaxes(
        title="Rentabilidad acumulada",
        ticksuffix="%",
        gridcolor="#dbe3eb",
        tickfont=dict(color="#334155", size=12),
        automargin=True,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False, "displayModeBar": False, "responsive": True},
    )

    st.markdown('<div class="section">Cartera</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Rentabilidad {period}",
        pct(portfolio_stats["return"]),
        delta=f'{portfolio_stats["return"] - benchmark_stats["return"]:+.2%} vs S&P 500',
    )
    c2.metric("Drawdown máximo", pct(portfolio_stats["dd"]))
    c3.metric("Rentabilidad / drawdown", ratio(portfolio_stats["ratio"]))

    st.markdown('<div class="section">S&P 500</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.metric(f"Rentabilidad {period}", pct(benchmark_stats["return"]))
    s2.metric("Drawdown máximo", pct(benchmark_stats["dd"]))
    s3.metric("Rentabilidad / drawdown", ratio(benchmark_stats["ratio"]))

    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        x=drawdown_pct.index,
        y=drawdown_pct["Cartera"],
        name="Drawdown cartera",
        mode="lines",
        line=dict(color="#8b1e3f", width=2.6),
        fill="tozeroy",
        fillcolor="rgba(139,30,63,0.16)",
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>Cartera</extra>",
    ))
    dd_fig.add_trace(go.Scatter(
        x=drawdown_pct.index,
        y=drawdown_pct["S&P 500"],
        name="Drawdown S&P 500",
        mode="lines",
        line=dict(color="#e07a3f", width=2.2),
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>S&P 500</extra>",
    ))
    dd_fig.add_hline(y=0, line_width=1, line_color="#94a3b8")
    common_layout(dd_fig, 420, f"Curva de drawdown {period}")
    dd_fig.update_yaxes(
        title="Drawdown",
        ticksuffix="%",
        gridcolor="#dbe3eb",
        tickfont=dict(color="#334155", size=12),
        automargin=True,
        rangemode="tozero",
    )
    st.plotly_chart(
        dd_fig,
        width="stretch",
        config={"displaylogo": False, "displayModeBar": False, "responsive": True},
    )

    st.markdown('<div class="section">Situación actual</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("Patrimonio actual", money(portfolio["Patrimonio"].iloc[-1]))
    p2.metric("Capital invertido", money(portfolio["Invertido"].iloc[-1]))
    p3.metric("Efectivo", money(portfolio["Efectivo"].iloc[-1]))

    year_start = pd.Timestamp(today.year, 1, 1)
    history = trades[(trades["BUY DATE"] >= year_start) & (trades["BUY DATE"] <= today)].copy()
    history["ESTADO"] = np.where(
        history["SELL DATE"].notna() & (history["SELL DATE"] <= today),
        "Cerrada",
        "Abierta",
    )
    history["FINAL"] = history["SELL DATE"].where(history["ESTADO"].eq("Cerrada"), today)
    history["DURACION"] = (history["FINAL"] - history["BUY DATE"]).dt.days
    history["RENTABILIDAD"] = history.apply(operation_return, axis=1, prices=prices, today=today)
    open_count = int(history["ESTADO"].eq("Abierta").sum())
    closed_count = int(history["ESTADO"].eq("Cerrada").sum())

    st.subheader(f"Histórico de operaciones {today.year}")
    st.caption(
        f"{len(history)} operaciones desde el 1 de enero · "
        f"{open_count} abiertas · {closed_count} cerradas"
    )
    wanted = [
        "TICKER", "COMPANY", "GICS SECTOR", "GICS INDUSTRY",
        "GICS SUB-INDUSTRY", "BUY DATE", "SELL DATE",
        "ESTADO", "DURACION", "RENTABILIDAD",
    ]
    operations = history[[column for column in wanted if column in history.columns]].copy()
    operations["ORDEN ESTADO"] = operations["ESTADO"].map({"Abierta": 0, "Cerrada": 1})
    operations = operations.sort_values(
        ["ORDEN ESTADO", "BUY DATE", "TICKER"],
        ascending=[True, False, True],
    ).drop(columns="ORDEN ESTADO")
    operations["BUY DATE"] = operations["BUY DATE"].dt.strftime("%d/%m/%Y")
    operations["SELL DATE"] = operations["SELL DATE"].dt.strftime("%d/%m/%Y").fillna("-")
    operations["ESTADO"] = operations["ESTADO"].map({
        "Abierta": "🟢 Abierta",
        "Cerrada": "🔴 Cerrada",
    })
    operations = operations.rename(columns={
        "TICKER": "Ticker",
        "COMPANY": "Compañía",
        "GICS SECTOR": "Sector GICS",
        "GICS INDUSTRY": "Industria GICS",
        "GICS SUB-INDUSTRY": "Subindustria GICS",
        "BUY DATE": "Fecha de compra",
        "SELL DATE": "Fecha de cierre",
        "ESTADO": "Estado",
        "DURACION": "Duración (días)",
        "RENTABILIDAD": "Rentabilidad",
    })

    def color_return(value):
        if pd.isna(value):
            return "font-weight:700; color:#64748b"
        if value > 0:
            return "font-weight:800; color:#0a8f55"
        if value < 0:
            return "font-weight:800; color:#d93025"
        return "font-weight:800; color:#475569"

    styled_operations = (
        operations.style
        .map(color_return, subset=["Rentabilidad"])
        .format({
            "Rentabilidad": lambda value: "-" if pd.isna(value) else f"{value:+.2%}",
            "Duración (días)": lambda value: "-" if pd.isna(value) else f"{int(value)} días",
        })
    )
    st.dataframe(styled_operations, width="stretch", hide_index=True)

    st.caption(
        f"Simulación con capital inicial de {money(INITIAL_CAPITAL)}, sin rebalanceo diario. "
        "En cada cambio mensual, el importe de las ventas se distribuye por igual "
        "entre las nuevas compras de esa fecha. Se permiten acciones fraccionarias."
    )

except Exception as exc:
    st.error(f"No se ha podido generar el dashboard: {exc}")
    st.exception(exc)
