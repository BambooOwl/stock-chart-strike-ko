"""
Streamlit Stock Chart Tool — v2.1 (completed)

This build rolls back to v1.9 and adds two date stamps (Generated & Latest
price) in the top‑right corner of every page. No other new features.
"""

import io
import datetime as dt
from itertools import islice
from typing import Iterable, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import streamlit as st
import yfinance as yf
from matplotlib.backends.backend_pdf import PdfPages

START_DATE = "2024-06-30"
LIGHT_GREEN = "#e6f4ea"
DARK_GREEN = "#238b45"
KO_BLUE = "#3182bd"
STRIKE_GREY = "#636363"
GRID_GREY = "#d0d0d0"

WATERMARK_TEXT = "UOB Kay Hian PWM Product Team"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean(col):
    col = str(col)
    return col if "." in col and col.count(".") == 1 else col.split(".")[0]

def fetch_prices(tickers: List[str], start: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        key = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        df = raw[key]
    else:
        cand = [c for c in ("Adj Close", "Close") if c in raw.columns]
        df = raw[cand] if cand else raw.to_frame(name=tickers[0])
    df.columns = [_clean(c) for c in df.columns]
    return df.ffill().dropna(how="all")


def long_name(ticker: str) -> str:
    try:
        return yf.Ticker(ticker).info.get("longName") or ticker
    except Exception:
        return ticker

def _header_y_positions(n):
    """
    Return y coordinates (top-to-bottom) for the four header lines.
    If only 1–2 charts, use looser spacing; otherwise keep compact.
    """
    if n < 3:                       # wide spacing
        return 0.97, 0.93, 0.89, 0.85
    else:                           # original compact block
        return 0.975, 0.955, 0.935, 0.915

# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _build_month_formatter(start_month: int, start_year: int):
    def _fmt(x, pos):
        dt_obj = mdates.num2date(x)
        if dt_obj.month == 1 or (dt_obj.month == start_month and dt_obj.year == start_year):
            return dt_obj.strftime("%b\n%Y")
        return dt_obj.strftime("%b")
    return FuncFormatter(_fmt)


def plot_group(df: pd.DataFrame,
               ko_ratio: Optional[float],
               strike_ratio: Optional[float],
               ki_ratio: Optional[float],
               gen_date: dt.date,
               latest_price_date: dt.date) -> plt.Figure:

    n = df.shape[1]

    HEADER_H = 2.0
    ROW_H = 3.5
    fig_h = HEADER_H + ROW_H * n
                   
    fig, axs = plt.subplots(n, 1, figsize=(7, fig_h))
    axs = [axs] if n == 1 else axs

    # ── Header block ─────────────────────────────────────────────────────────
    ratio_strike = f"{strike_ratio:.0%}" if strike_ratio else "NA"
    ratio_ko     = f"{ko_ratio:.0%}"    if ko_ratio     else "NA"
    ratio_ki     = f"{ki_ratio:.0%}"    if ki_ratio     else "NA"

    y_title, y_ratios, y_source, y_credit = _header_y_positions(n)
                   
    fig.text(0.01, y_title, "Stock Chart Tool", color='#c33b31', fontsize=12, fontweight="bold", va="top", ha="left")
    fig.text(0.01, y_ratios, f"Strike = {ratio_strike},  KO = {ratio_ko},  KI = {ratio_ki}",
             fontsize=11, fontweight="bold", va="top", ha="left")
    fig.text(0.01, y_source, "Source: Yahoo Finance, all data as indicative only.", fontsize=8, fontstyle="italic", va="top", ha="left")
    fig.text(0.01, y_credit, "Developed by Kit / Shi Jie,   Developed for UOB Kay Hian Private Wealth Research", fontsize=8,
             fontstyle="italic", va="top", ha="left")
    
    fig.text(0.99, 0.94, f"Generated: {gen_date:%d %b %Y}", fontsize=9, va="top", ha="right")
    fig.text(0.99, 0.92, f"Latest price: {latest_price_date:%d %b %Y}", fontsize=9, va="top", ha="right")

    fig.subplots_adjust(top=y_credit - 0.02, right=0.84, hspace=0.45, left=0.06)
                   
    for ax, ticker in zip(axs, df.columns):
        series = df[ticker]
        base_price = series.iloc[-1]

        ko = base_price * ko_ratio if ko_ratio else None
        strike = base_price * strike_ratio if strike_ratio else None
        ki = base_price * ki_ratio if ki_ratio else None

        # Y‑limits & fill baseline
        upper = max(filter(None, [series.max(), ko, ki]))
        lower = min(filter(None, [series.min(), strike, ki]))
        span  = upper - lower if upper != lower else abs(upper) * 0.1 or 1
        y_min = lower - 0.15 * span
        ax.set_ylim(y_min, upper + 0.15 * span)

        ax.fill_between(series.index, series, y_min, color=LIGHT_GREEN, alpha=0.4)
        ax.plot(series.index, series, color=DARK_GREEN, linewidth=1.5)

        if strike is not None:
            ax.axhline(strike, color=STRIKE_GREY, linewidth=1)
        if ko is not None:
            ax.axhline(ko, color=KO_BLUE, linewidth=1)
        if ki is not None:
            ax.axhline(ki, color="red", linewidth=1, linestyle="--")

        # Watermark
        ax.text(0.5, 0.5, WATERMARK_TEXT, transform=ax.transAxes,
                fontsize=12, color="#999999", alpha=0.12, ha="center", va="center",
                rotation=0)

        # Hide frame
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

        ax.grid(True, linestyle=":", linewidth=0.4, color=GRID_GREY, alpha=0.8)
        ax.set_title(f"{long_name(ticker)} ({ticker}), Price: {base_price:.2f}", fontsize=10, pad=8)
        ax.tick_params(labelsize=8)

        ax.set_xlim(series.index[0], series.index[-1])
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(_build_month_formatter(series.index[0].month, series.index[0].year))
        ax.tick_params(axis="x", rotation=0, pad=2)

        label_x = 1.01
        trans    = ax.get_yaxis_transform()
        if ko is not None:
            ax.text(label_x, ko, f"KO {ko:.2f}", color=KO_BLUE, va="center", ha="left", fontsize=8, transform=trans)
        if strike is not None:
            ax.text(label_x, strike, f"Strike {strike:.2f}", color=STRIKE_GREY, va="center", ha="left", fontsize=8, transform=trans)
        if ki is not None:
            ax.text(label_x, ki, f"KI {ki:.2f}", color="red", va="center", ha="left", fontsize=8, transform=trans)

    return fig

# ----------------------------------------------------------------------------

def chunk(iterable: Iterable, size: int):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Stock Chart Tool", layout="centered")

    t_in         = st.text_input("Tickers (comma separated)", "MSFT,NVDA,AAPL,AMZN,GOOGL,META,TSLA")
    ko_ratio     = st.number_input("KO ratio (blank = NA)", value=1.05, step=0.01)
    strike_ratio = st.number_input("Strike ratio (blank = NA)", value=0.80, step=0.01)
    ki_ratio     = st.number_input("KI ratio (optional)", value=0.0, step=0.01)
    use_ki       = st.checkbox("Enable KI", value=False)

    if st.button("Generate charts"):
        tickers = [s.strip().upper() for s in t_in.split(',') if s.strip()]
        if not tickers:
            st.error("Please enter at least one ticker.")
            st.stop()
        try:
            data = fetch_prices(tickers, START_DATE)
        except Exception as e:
            st.error(f"Download error: {e}")
            st.stop()

        gen_date          = dt.date.today()
        latest_price_date = max(data.index).date()

        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            for group in chunk(tickers, 4):
                fig = plot_group(data[group],
                                 ko_ratio or None,
                                 strike_ratio or None,
                                 ki_ratio if use_ki else None,
                                 gen_date,
                                 latest_price_date)
                st.pyplot(fig)
                pdf.savefig(fig)
                plt.close(fig)

        buf.seek(0)
        st.download_button("Download PDF", buf, file_name=f"Mag7 {gen_date:%d %b}.pdf", mime="application/pdf")


if __name__ == "__main__":
    main()
