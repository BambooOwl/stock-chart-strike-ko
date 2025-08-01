"""
Streamlit Stock Chart Tool — v2.1 (completed)
"""

import io
import datetime as dt
from itertools import islice
from typing import Iterable, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
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
    end = (dt.date.today() + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
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

def _header_y_positions(n: int):
    """
    Returns y coordinates, top to bottom.
    Gap between title ↔ ratios ↔ source is dynamic,
    but source ↔ credit is a tight 0.01.
    """
    base = 0.97            # title line
    if n == 1:
        g1 = 0.04
        g2 = 0.03
    elif n == 2:
        g1, g2 = 0.03, 0.025
    elif n == 3:
        g1, g2 = 0.025, 0.02
    else:                  # n >= 4
        g1, g2 = 0.02, 0.02

    y_title = base
    y_ratios = base - g1
    y_source = y_ratios -g2
    y_credit = y_source - 0.01
    return y_title, y_ratios, y_source, y_credit

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
               ko_pct, strike_pct, ki_pct,
               coupon_pa, tenor_months, stepdown_pct,
               gen_date: dt.date,
               latest_price_date: dt.date) -> plt.Figure:

    n = df.shape[1]

    HEADER_H = 2.0
    ROW_H = 2.75
    fig_h = HEADER_H + ROW_H * n
                   
    fig = plt.figure(figsize=(7, fig_h))

    # outer grid: row0 for header (1 unit tall), row1 for charts (n units tall)
    outer = GridSpec(2, 1,
                     height_ratios=[HEADER_H, ROW_H * n],
                     hspace=-0.06,        # Space between Top Wording and Bottom Chart (The Empty Zone)
                     figure=fig)

    # --- HEADER AXES (row 0) ---
    axh = fig.add_subplot(outer[0])
    axh.axis("off")   # hide the spines/ticks

    # place your four header lines here in axh coordinates
    axh.text(-0.05, 0.85, "Stock Chart Tool",
             fontsize=12, fontweight="bold", color="#c33b31",
             transform=axh.transAxes, ha="left", va="center")

    axh.text(-0.05, 0.75,
             f"Strike = {strike_pct:.2f}%,  KO = {ko_pct:.2f}%,  KI = {ki_pct:.2f}" if ki_pct else
             f"Strike = {strike_pct:.2f}%,  KO = {ko_pct:.2f}%,  KI = NA",
             fontsize=11, fontweight="bold",
             transform=axh.transAxes, ha="left", va="center")

    axh.text(-0.05, 0.65,
             f"Coupon p.a. = {coupon_pa:.2f}%,  Tenor (months) = {tenor_months:.0f}, Stepdown (%) = {stepdown_pct:.2f}" if stepdown_pct else
             f"Coupon p.a. = {coupon_pa:.2f}%,  Tenor (months) = {tenor_months:.0f}, Stepdown (%) = NA",
             fontsize=11, fontweight="bold",
             transform=axh.transAxes, ha="left", va="center")

    axh.text(-0.05, 0.55,
             "Source: Yahoo Finance, all data as indicative only.",
             fontsize=8, fontstyle="italic",
             transform=axh.transAxes, ha="left", va="center")

    axh.text(-0.05, 0.45,
             "Developed by Kit,   Developed for UOB Kay Hian Private Wealth Research",
             fontsize=8, fontstyle="italic",
             transform=axh.transAxes, ha="left", va="center")

    axh.text(1.12, 0.85, f"Generated: {gen_date:%d %b %Y}",
             fontsize=9, transform=axh.transAxes,
             ha="right", va="center")

    axh.text(1.12, 0.75, f"Latest price: {latest_price_date:%d %b %Y}",
             fontsize=9, transform=axh.transAxes,
             ha="right", va="center")

    # --- CHART AXES (row 1 subdivided into n sub-rows) ---
    inner = outer[1].subgridspec(n, 1, hspace=0.4)
    axs   = [fig.add_subplot(inner[i, 0]) for i in range(n)]
                   
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
        ax.set_title(f"{long_name(ticker)} ({ticker}), Price: {series.iloc[-1]:.2f}", fontsize=10, pad=8)
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

    col1, col2, col3 = st.columns(3)
    with col1:
        ko_pct     = st.number_input("KO (%)", value=105.50, step=0.01, format="%.2f", min_value=0.0)
    with col2:
        strike_pct = st.number_input("Strike (%)", value=80.50, step=0.01, format="%.2f", min_value=0.0)
    with col3:
        ki_pct     = st.number_input("KI (%)", value=55.55, step=0.01, format="%.2f", min_value=0.0)

    col4, col5, col6 = st.columns(3)
    with col4:
        coupon_pa = st.number_input("Coupon p.a. (%)", value=12.00, step=0.01, format="%.2f", min_value=0.0)
    with col5:
        tenor_months = st.number_input("Tenor (months)", value=3.00, step=0.01, format="%.2f", min_value=0.0)
    with col6:
        stepdown_pct = st.number_input("Stepdown (%)", value=1.00, step=0.01, format="%.2f", min_value=0.0)
    
    use_ki       = st.checkbox("Enable KI", value=True)

    ko_ratio     = ko_pct / 100 if ko_pct else None
    strike_ratio = strike_pct / 100 if strike_pct else None
    ki_ratio     = ki_pct / 100 if use_ki and ki_pct else None
    
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
                                 ko_pct, strike_pct, ki_pct,
                                 coupon_pa, tenor_months, stepdown_pct,
                                 gen_date,
                                 latest_price_date)
                st.pyplot(fig)
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)

        buf.seek(0)
        st.download_button("Download PDF", buf, file_name=f"Mag7 {gen_date:%d %b}.pdf", mime="application/pdf")


if __name__ == "__main__":
    main()
