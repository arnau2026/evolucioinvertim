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
    df.columns = [str(c).strip().upper() for c in df.columns]
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
    if (df["SELL DATE"].notna() & (df["SELL DATE"] < df["BUY DATE"])).any():
        raise ValueError("Hay alguna fecha de venta anterior a la compra.")
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


def build_returns(trades, prices, start, today):
    tickers = [column for column in prices.columns if column != BENCHMARK]
    returns = prices[tickers].ffill().pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    active = pd.DataFrame(False, index=returns.index, columns=tickers)

    for _, trade in trades.iterrows():
        ticker = trade["TICKER"]
        if ticker not in active.columns:
            continue
        sell = today if pd.isna(trade["SELL DATE"]) else min(trade["SELL DATE"], today)
        active.loc[(active.index > trade["BUY DATE"]) & (active.index <= sell), ticker] = True

    valid = active & returns.notna()
    count = valid.sum(axis=1)
    portfolio = returns.where(valid).sum(axis=1).div(count.replace(0, np.nan)).fillna(0)
    spy = prices[BENCHMARK].ffill().pct_change(fill_method=None).fillna(0)
    daily = pd.concat(
        [portfolio.rename("Cartera"), spy.rename("S&P 500")], axis=1
    ).loc[start:today]
    return daily


def period_start(period, today):
    if period == "YTD":
        return pd.Timestamp(today.year, 1, 1)
    return (today - PERIODS[period]).normalize()


def operation_return(row, prices, today):
    ticker = row["TICKER"]
    if ticker not in prices.columns:
        return np.nan
    series = prices[ticker].dropna()
    if series.empty:
        return np.nan

    buy_candidates = series.loc[series.index >= row["BUY DATE"]]
    if buy_candidates.empty:
        return np.nan
    buy_price = buy_candidates.iloc[0]

    end_date = today if pd.isna(row["SELL DATE"]) or row["SELL DATE"] > today else row["SELL DATE"]
    sell_candidates = series.loc[series.index <= end_date]
    if sell_candidates.empty or buy_price <= 0:
        return np.nan
    return sell_candidates.iloc[-1] / buy_price - 1


def stats_for(daily):
    result = {}
    for column in daily.columns:
        growth = (1 + daily[column]).cumprod()
        drawdown = growth / growth.cummax() - 1
        total_return = growth.iloc[-1] - 1
        max_drawdown = drawdown.min()
        result[column] = {
            "return": total_return,
            "dd": max_drawdown,
            "ratio": total_return / abs(max_drawdown) if max_drawdown < 0 else np.nan,
        }
    return result


def pct(value):
    return "N/D" if pd.isna(value) else f"{value:+.2%}"


def ratio(value):
    return "N/D" if pd.isna(value) or np.isinf(value) else f"{value:.2f}x"


def common_layout(fig, height, title):
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=15, t=85, b=35),
        title=dict(text=title, x=0.01, font=dict(color="#172033", size=18)),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#334155", size=13),
        xaxis=dict(
            showgrid=False, linecolor="#94a3b8",
            tickfont=dict(color="#334155", size=12), automargin=True,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(color="#334155", size=12),
            bgcolor="rgba(255,255,255,0.85)",
        ),
        hoverlabel=dict(
            bgcolor="#111827", bordercolor="#111827",
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
    calculation_start = min(
        pd.Timestamp(today.year, 1, 1),
        today - pd.DateOffset(months=6),
        trades["BUY DATE"].min(),
    )
    tickers = tuple(sorted(set(trades["TICKER"]) | {BENCHMARK}))
    prices = get_prices(
        tickers,
        (calculation_start - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        (today + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
    )
    if prices.empty or BENCHMARK not in prices.columns:
        st.error("No se han podido descargar los precios.")
        st.stop()

    missing = sorted(set(trades["TICKER"]) - set(prices.columns))
    if missing:
        st.warning("Sin precios para: " + ", ".join(missing))
        trades = trades[~trades["TICKER"].isin(missing)]

    daily_all = build_returns(trades, prices, calculation_start, today)

    period_col, spacer_col, refresh_col = st.columns([5.0, 3.5, 1.5], vertical_alignment="center")
    with period_col:
        period = st.radio(
            "Periodo", ["1S", "1M", "3M", "6M", "YTD"],
            index=4, horizontal=True, label_visibility="collapsed",
        )
    with refresh_col:
        if st.button("↻ Actualizar", type="primary", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    daily = daily_all.loc[daily_all.index >= period_start(period, today)].copy()
    if daily.empty:
        raise ValueError("No hay sesiones en el periodo seleccionado.")

    growth = (1 + daily).cumprod()
    curve = growth * 100 - 100
    drawdown_curve = (growth / growth.cummax() - 1) * 100
    stats = stats_for(daily)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve.index, y=curve["Cartera"], name="Cartera", mode="lines",
        line=dict(color="#087f8c", width=3),
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>Cartera</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=curve.index, y=curve["S&P 500"], name="S&P 500 (SPY)", mode="lines",
        line=dict(color="#e07a3f", width=2.6),
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>S&P 500</extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#9aa8b6")
    common_layout(
        fig, 555,
        f"Rentabilidad acumulada {period} · Última sesión: {curve.index[-1]:%d/%m/%Y}",
    )
    fig.update_yaxes(
        title=dict(text="Rentabilidad acumulada", font=dict(color="#334155", size=13)),
        ticksuffix="%", gridcolor="#dbe3eb", linecolor="#94a3b8",
        tickfont=dict(color="#334155", size=12), automargin=True,
    )
    st.plotly_chart(
        fig, width="stretch",
        config={"displaylogo": False, "displayModeBar": False, "responsive": True},
    )

    st.markdown('<div class="section">Cartera</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Rentabilidad {period}", pct(stats["Cartera"]["return"]),
        delta=f'{stats["Cartera"]["return"] - stats["S&P 500"]["return"]:+.2%} vs S&P 500',
    )
    c2.metric("Drawdown máximo", pct(stats["Cartera"]["dd"]))
    c3.metric("Rentabilidad / drawdown", ratio(stats["Cartera"]["ratio"]))

    st.markdown('<div class="section">S&P 500</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.metric(f"Rentabilidad {period}", pct(stats["S&P 500"]["return"]))
    s2.metric("Drawdown máximo", pct(stats["S&P 500"]["dd"]))
    s3.metric("Rentabilidad / drawdown", ratio(stats["S&P 500"]["ratio"]))

    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        x=drawdown_curve.index,
        y=drawdown_curve["Cartera"],
        name="Drawdown cartera",
        mode="lines",
        line=dict(color="#8b1e3f", width=2.6),
        fill="tozeroy",
        fillcolor="rgba(139,30,63,0.16)",
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>Cartera</extra>",
    ))
    dd_fig.add_trace(go.Scatter(
        x=drawdown_curve.index,
        y=drawdown_curve["S&P 500"],
        name="Drawdown S&P 500",
        mode="lines",
        line=dict(color="#e07a3f", width=2.2),
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>S&P 500</extra>",
    ))
    dd_fig.add_hline(y=0, line_width=1, line_color="#94a3b8")
    common_layout(dd_fig, 420, f"Curva de drawdown {period}")
    dd_fig.update_yaxes(
        title=dict(text="Drawdown", font=dict(color="#334155", size=13)),
        ticksuffix="%", gridcolor="#dbe3eb", linecolor="#94a3b8",
        tickfont=dict(color="#334155", size=12), automargin=True,
        rangemode="tozero",
    )
    st.plotly_chart(
        dd_fig, width="stretch",
        config={"displaylogo": False, "displayModeBar": False, "responsive": True},
    )

    year_start = pd.Timestamp(today.year, 1, 1)
    history = trades[(trades["BUY DATE"] >= year_start) & (trades["BUY DATE"] <= today)].copy()
    history["ESTADO"] = np.where(
        history["SELL DATE"].notna() & (history["SELL DATE"] <= today),
        "Cerrada", "Abierta",
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
        "GICS SUB-INDUSTRY", "BUY DATE", "SELL DATE", "ESTADO",
        "DURACION", "RENTABILIDAD",
    ]
    operations = history[[column for column in wanted if column in history.columns]].copy()
    operations["ORDEN ESTADO"] = operations["ESTADO"].map({"Abierta": 0, "Cerrada": 1})
    operations = operations.sort_values(
        ["ORDEN ESTADO", "BUY DATE", "TICKER"], ascending=[True, False, True]
    ).drop(columns="ORDEN ESTADO")
    operations["BUY DATE"] = operations["BUY DATE"].dt.strftime("%d/%m/%Y")
    operations["SELL DATE"] = operations["SELL DATE"].dt.strftime("%d/%m/%Y").fillna("-")
    operations["ESTADO"] = operations["ESTADO"].map({
        "Abierta": "🟢 Abierta", "Cerrada": "🔴 Cerrada"
    })
    operations = operations.rename(columns={
        "TICKER": "Ticker", "COMPANY": "Compañía",
        "GICS SECTOR": "Sector GICS", "GICS INDUSTRY": "Industria GICS",
        "GICS SUB-INDUSTRY": "Subindustria GICS",
        "BUY DATE": "Fecha de compra", "SELL DATE": "Fecha de cierre",
        "ESTADO": "Estado", "DURACION": "Duración (días)",
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
        "Cartera equiponderada y rebalanceada diariamente entre posiciones activas. "
        "Precios ajustados por dividendos y splits."
    )

except Exception as exc:
    st.error(f"No se ha podido generar el dashboard: {exc}")
    st.exception(exc)
