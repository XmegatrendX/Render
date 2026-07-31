import io
import os
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta   # ✅ добавлен timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Настройки ---
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable is not set!")

URL     = "https://smartmoney-bot-ilqm.onrender.com"
CHAT_ID = int(os.getenv("CHAT_ID", "0"))

# --- Telegram Bot ---
bot_app = ApplicationBuilder().token(TOKEN).build()

FUTURES = {
    'gc': 'GC=F', 'cl': 'CL=F', 'pl': 'PL=F',
    '6e': '6E=F', '6j': '6J=F', 'dx': 'DX=F'
}

# ────────────────────────────────────────────────
# Комбинированные пары: /CGC (GC+PL), /CCL (CL+Brent)
# ────────────────────────────────────────────────
COMBINED = {
    'cgc': {
        'sym1': FUTURES['gc'], 'label1': 'GC', 'color1': 'red',
        'sym2': FUTURES['pl'], 'label2': 'PL', 'color2': 'black',
        'title': 'GC vs PL — Volume Stress / Participation Index',
    },
    'ccl': {
        'sym1': FUTURES['cl'], 'label1': 'CL',    'color1': 'red',
        'sym2': 'BZ=F',        'label2': 'BRENT', 'color2': 'black',
        'title': 'CL vs BRENT — Volume Stress / Participation Index',
    },
}


# ────────────────────────────────────────────────
# Лунные перигеи
# ────────────────────────────────────────────────
def get_lunar_perigees(days_back: int = 175) -> list:
    """Возвращает даты перигеев за последние days_back дней + один следующий."""
    try:
        import ephem
    except ImportError:
        return []
    perigees  = []
    start     = ephem.Date(datetime.now() - pd.Timedelta(days=days_back + 30))
    end       = ephem.Date(datetime.now() + pd.Timedelta(days=35))
    moon      = ephem.Moon()
    cutoff    = pd.Timestamp(datetime.now() - pd.Timedelta(days=days_back))
    now_ts    = pd.Timestamp(datetime.now())
    date      = start
    prev_dist = None
    prev_date = None
    while date < end:
        moon.compute(date)
        dist = moon.earth_distance
        if prev_dist is not None and prev_dist < dist:
            lo, hi = prev_date, date
            for _ in range(30):
                m1 = lo + (hi - lo) / 3
                m2 = lo + (hi - lo) * 2 / 3
                moon.compute(m1); d1 = moon.earth_distance
                moon.compute(m2); d2 = moon.earth_distance
                if d1 < d2:
                    hi = m2
                else:
                    lo = m1
            perigee = (lo + hi) / 2
            p = pd.Timestamp(ephem.Date(perigee).datetime())
            if (not perigees or (p - perigees[-1]).days > 20) and p >= cutoff:
                perigees.append(p)
                if p > now_ts:
                    break
        prev_dist = dist
        prev_date = date
        date += 0.5
    return perigees


# ────────────────────────────────────────────────
# Smart Money Flow
# ────────────────────────────────────────────────
def smart_money_flow(symbol, days=175):
    df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)
    if df is None or len(df) < 20:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df['Vol_Pct'] = df['Volume'].rolling(20).apply(
        lambda x: (x[:-1] < x[-1]).sum() / (len(x) - 1) * 100, raw=True
    )

    raw_trend       = df['Volume'].rolling(5).mean() / (df['Volume'].rolling(20).mean() + 1e-8) - 1
    df['Vol_Trend'] = (raw_trend.clip(-1, 1) + 1) * 50

    raw_acc         = df['Close'].pct_change().diff().fillna(0)
    df['Price_Acc'] = (raw_acc.clip(-0.03, 0.03) / 0.03 + 1) * 50

    df['Signal'] = (
        0.7 * df['Vol_Pct'] +
        0.2 * df['Vol_Trend'] +
        0.1 * df['Price_Acc']
    ).fillna(50)

    vol_std   = df['Volume'].rolling(20).std() / (df['Volume'].rolling(20).mean() + 1e-8)
    span_vals = (3 + (1 - vol_std.clip(0, 1)) * 7).fillna(5).round().astype(int).values
    sig_vals  = df['Signal'].values
    result    = np.zeros(len(sig_vals))
    result[0] = sig_vals[0]
    for i in range(1, len(sig_vals)):
        alpha     = 2.0 / (span_vals[i] + 1.0)
        result[i] = alpha * sig_vals[i] + (1 - alpha) * result[i - 1]

    vol_z     = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
    mfm       = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8)
    smart_vol = (mfm * df['Volume'] * vol_z.clip(lower=1)).fillna(0)
    smart_ad  = smart_vol.cumsum().ewm(span=5).mean().values
    ad_min, ad_max = smart_ad.min(), smart_ad.max()
    if ad_max - ad_min > 1e-8:
        smart_ad_pct = (smart_ad - ad_min) / (ad_max - ad_min) * 100
    else:
        smart_ad_pct = np.full(len(smart_ad), 50.0)
    smart_ad_s  = pd.Series(smart_ad_pct, index=df.index)
    center      = smart_ad_s.rolling(200, min_periods=50).median()
    direction   = np.sign(smart_ad_s - center).fillna(0).values

    flow_signed  = 50 + (result - 50) * direction
    flow_clipped = np.clip(flow_signed, 0, 100)

    flow_s   = pd.Series(flow_clipped)
    envelope = np.where(
        flow_s >= 50,
        flow_s.rolling(5, min_periods=1).max(),
        flow_s.rolling(5, min_periods=1).min()
    )
    smoothed = pd.Series(envelope).ewm(span=3).mean().values

    df['Flow'] = smoothed
    return df


# ────────────────────────────────────────────────
# RSX Джурика
# ────────────────────────────────────────────────
def calculate_rsx(series: pd.Series, length: int = 9) -> pd.Series:
    src = series.values.astype(float)
    n   = len(src)
    rsx = np.zeros(n)
    f8  = np.zeros(n); f10 = np.zeros(n); v8 = np.zeros(n)
    f18 = 3.0 / (length + 2.0); f20 = 1.0 - f18
    f28 = np.zeros(n); f30 = np.zeros(n)
    f38 = np.zeros(n); f40 = np.zeros(n)
    f48 = np.zeros(n); f50 = np.zeros(n)
    f58 = np.zeros(n); f60 = np.zeros(n)
    f68 = np.zeros(n); f70 = np.zeros(n)
    f78 = np.zeros(n); f80 = np.zeros(n)
    f88 = np.zeros(n); f90 = np.zeros(n)

    for i in range(n):
        f8[i]  = 100.0 * src[i]
        f10[i] = f8[i - 1] if i > 0 else 0.0
        v8[i]  = f8[i] - f10[i]
        f28[i] = f20 * (f28[i-1] if i > 0 else 0.0) + f18 * v8[i]
        f30[i] = f18 * f28[i] + f20 * (f30[i-1] if i > 0 else 0.0)
        vC     = f28[i] * 1.5 - f30[i] * 0.5
        f38[i] = f20 * (f38[i-1] if i > 0 else 0.0) + f18 * vC
        f40[i] = f18 * f38[i] + f20 * (f40[i-1] if i > 0 else 0.0)
        v10    = f38[i] * 1.5 - f40[i] * 0.5
        f48[i] = f20 * (f48[i-1] if i > 0 else 0.0) + f18 * v10
        f50[i] = f18 * f48[i] + f20 * (f50[i-1] if i > 0 else 0.0)
        v14    = f48[i] * 1.5 - f50[i] * 0.5
        f58[i] = f20 * (f58[i-1] if i > 0 else 0.0) + f18 * abs(v8[i])
        f60[i] = f18 * f58[i] + f20 * (f60[i-1] if i > 0 else 0.0)
        v18    = f58[i] * 1.5 - f60[i] * 0.5
        f68[i] = f20 * (f68[i-1] if i > 0 else 0.0) + f18 * v18
        f70[i] = f18 * f68[i] + f20 * (f70[i-1] if i > 0 else 0.0)
        v1C    = f68[i] * 1.5 - f70[i] * 0.5
        f78[i] = f20 * (f78[i-1] if i > 0 else 0.0) + f18 * v1C
        f80[i] = f18 * f78[i] + f20 * (f80[i-1] if i > 0 else 0.0)
        v20    = f78[i] * 1.5 - f80[i] * 0.5
        f88[i] = length - 1 if (i > 0 and f90[i-1] == 0 and length - 1 >= 5) else 5
        f90[i] = (
            1 if i == 0 or f90[i-1] == 0
            else f88[i] + 1 if f88[i] <= f90[i-1]
            else f90[i-1] + 1
        )
        f0     = 1 if (f88[i] >= f90[i] and f8[i] != f10[i]) else 0
        if f88[i] == f90[i] and f0 == 0:
            f90[i] = 0
        v4     = (v14 / v20 + 1.0) * 50.0 if (f88[i] < f90[i] and abs(v20) > 1e-8) else 50.0
        rsx[i] = max(0.0, min(100.0, v4))

    return pd.Series(rsx, index=series.index)


# ────────────────────────────────────────────────
# Общие хелперы для графиков (перигеи + таблица фаз)
# ────────────────────────────────────────────────
def draw_perigees(ax1, ax2, perigees, now_ts):
    for p in perigees:
        p_n = p.tz_localize(None) if p.tzinfo is not None else p
        if p_n <= now_ts:
            ax1.axvline(p_n, color='red', linestyle='--', linewidth=0.9, alpha=0.6, zorder=5)
            ax2.axvline(p_n, color='red', linestyle='--', linewidth=0.9, alpha=0.6, zorder=5)
        else:
            ax1.axvline(p_n, color='red', linestyle='--', linewidth=1.4, alpha=0.95, zorder=5)
            ax2.axvline(p_n, color='red', linestyle='--', linewidth=1.4, alpha=0.95, zorder=5)
            ax1.text(p_n, 93, f"↓ {p.strftime('%d.%m')}", color='red',
                     fontsize=7.5, ha='center', fontweight='bold', zorder=6,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='crimson', alpha=0.9))
            break


def draw_phase_table(ax3):
    phases = [
        ("Accumulation",  "Накопление",      "30–50",   "40–60",  "Участие начинает появляться",           "Формирование базы, подготовка режима",  "#4169E1"),
        ("Expansion",     "Экспансия",       "> 60–70", "> 70",   "Резкое ускорение участия",              "Включился импульс режима",              "#32CD32"),
        ("Trend",         "Тренд",           "> 70",    "50–70",  "Участие держится стабильно",            "Режим устойчив, идёт протяжка",         "#006400"),
        ("Distribution",  "Распределение",   "> 70",    "↓ LH",   "Импульс участия слабеет",               "Смарт-деньги выгружаются",              "#FF8C00"),
        ("Balance",       "Сжатие/Баланс",   "45–55",   "40–60",  "Нет явного режима",                     "Переходная зона",                       "#888888"),
        ("Collapse",      "Коллапс",         "< 40",    "< 30",   "Резкий выход участия вниз",             "Смена режима / ликвидация",             "#DC143C"),
        ("Bear Exp.",     "Медвежья эксп.",  "< 40",    "< 30",   "Ускорение вниз",                        "Активный продавец",                     "#8B0000"),
        ("Bear Trend",    "Медвежий тренд",  "< 30",    "30–50",  "Давление удерживается",                 "Стабильный медвежий режим",             "#660000"),
    ]

    col_labels = ["Фаза", "Рус. название", "Flow", "RSX(Flow)", "Что происходит", "Что это значит"]
    col_widths  = [0.11,   0.13,            0.07,   0.07,        0.32,             0.30]
    row_h    = 0.105
    header_y = 0.95

    x = 0.0
    for label, w in zip(col_labels, col_widths):
        ax3.add_patch(plt.Rectangle((x, header_y - 0.04), w, 0.09,
                                    transform=ax3.transAxes,
                                    fc='#1a1a2e', ec='none', clip_on=False))
        ax3.text(x + w/2, header_y, label,
                 ha='center', va='center', fontsize=7.5, fontweight='bold',
                 color='white', transform=ax3.transAxes)
        x += w

    for row_i, (eng, rus, flow_r, rsx_r, what, meaning, color) in enumerate(phases):
        y  = header_y - 0.04 - (row_i + 1) * row_h
        bg = '#f4f4f4' if row_i % 2 == 0 else '#ffffff'
        row_data = [eng, rus, flow_r, rsx_r, what, meaning]
        x = 0.0
        for col_i, (val, w) in enumerate(zip(row_data, col_widths)):
            ax3.add_patch(plt.Rectangle((x, y - row_h*0.45), w, row_h*0.9,
                                        transform=ax3.transAxes,
                                        fc=bg, ec='#dddddd', linewidth=0.4,
                                        clip_on=False))
            if col_i == 0:
                ax3.add_patch(plt.Rectangle((x, y - row_h*0.45), 0.005, row_h*0.9,
                                            transform=ax3.transAxes,
                                            fc=color, ec='none', clip_on=False))
            ax3.text(x + w/2, y, val,
                     ha='center', va='center', fontsize=6.8,
                     transform=ax3.transAxes, color='#111111')
            x += w


def _localize_naive(df_or_series):
    """Убирает tz из индекса, если он есть."""
    if hasattr(df_or_series.index, 'tz') and df_or_series.index.tz is not None:
        df_or_series.index = df_or_series.index.tz_localize(None)
    return df_or_series


# ────────────────────────────────────────────────
# График: Flow + RSX + таблица фаз (один инструмент)
# ────────────────────────────────────────────────
def make_chart(df, symbol):
    import matplotlib.gridspec as gridspec

    rsx      = calculate_rsx(df['Flow'], length=9)
    perigees = get_lunar_perigees(175)
    now_ts   = pd.Timestamp(datetime.now()).tz_localize(None)

    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
        rsx.index = rsx.index.tz_localize(None)

    fig = plt.figure(figsize=(12, 11))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1, 0.85], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')

    ax1.plot(df.index, df['Flow'], label="Participation Index", linewidth=2, color='navy')
    ax1.axhline(85, color='red',   linestyle='--', linewidth=1, label='Overbought (85)')
    ax1.axhline(15, color='green', linestyle='--', linewidth=1, label='Oversold (15)')
    ax1.axhline(50, color='gray',  linestyle='-',  alpha=0.4)
    ax1.fill_between(df.index, 85, df['Flow'].clip(lower=85), alpha=0.15, color='red')
    ax1.fill_between(df.index, df['Flow'].clip(upper=15), 15, alpha=0.15, color='blue')
    ax1.set_title(f"{symbol} — Volume Stress / Participation Index", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(alpha=0.3)

    try:
        last_flow = float(df['Flow'].iloc[-1])
        last_date = df.index[-1]
        ax1.annotate(
            f"{last_date.strftime('%d.%m.%Y')}\n{last_flow:.1f}%",
            xy=(last_date, last_flow),
            xytext=(-60, 15), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
            fontsize=9, fontweight='bold',
        )
    except Exception:
        pass

    ax2.plot(df.index, rsx, label="RSX(9)", linewidth=1.5, color='orange')
    ax2.axhline(70, color='red',   linestyle='--', linewidth=1)
    ax2.axhline(30, color='green', linestyle='--', linewidth=1)
    ax2.axhline(50, color='gray',  linestyle='-',  alpha=0.4)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSX(9)', fontsize=9)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(alpha=0.3)

    try:
        last_rsx_date = rsx.index[-1]
        last_rsx_val  = float(rsx.iloc[-1])
        ax2.annotate(
            f"{last_rsx_date.strftime('%d.%m.%Y')}\nRSX: {last_rsx_val:.1f}",
            xy=(last_rsx_date, last_rsx_val),
            xytext=(-60, 15), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
            fontsize=9, fontweight='bold',
        )
    except Exception:
        pass

    draw_perigees(ax1, ax2, perigees, now_ts)
    draw_phase_table(ax3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


# ────────────────────────────────────────────────
# График: Flow + RSX + таблица фаз (два инструмента combined)
# ────────────────────────────────────────────────
def make_combined_chart(df1, df2, cfg):
    """cfg: словарь с ключами label1, color1, label2, color2, title."""
    import matplotlib.gridspec as gridspec

    label1, color1 = cfg['label1'], cfg['color1']
    label2, color2 = cfg['label2'], cfg['color2']

    df1 = df1.copy()
    df2 = df2.copy()
    rsx1 = calculate_rsx(df1['Flow'], length=9)
    rsx2 = calculate_rsx(df2['Flow'], length=9)
    perigees = get_lunar_perigees(175)
    now_ts   = pd.Timestamp(datetime.now()).tz_localize(None)

    _localize_naive(df1); _localize_naive(rsx1)
    _localize_naive(df2); _localize_naive(rsx2)

    fig = plt.figure(figsize=(12, 11))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1, 0.85], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')

    ax1.plot(df1.index, df1['Flow'], label=f"{label1} — Participation Index", linewidth=2, color=color1)
    ax1.plot(df2.index, df2['Flow'], label=f"{label2} — Participation Index", linewidth=2, color=color2)
    ax1.axhline(85, color='red',   linestyle='--', linewidth=1, label='Overbought (85)')
    ax1.axhline(15, color='green', linestyle='--', linewidth=1, label='Oversold (15)')
    ax1.axhline(50, color='gray',  linestyle='-',  alpha=0.4)
    ax1.set_title(cfg['title'], fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(alpha=0.3)

    flow_offsets = [(-70, 25), (-70, -40)]
    for (df, color, label), off in zip(((df1, color1, label1), (df2, color2, label2)), flow_offsets):
        try:
            last_flow = float(df['Flow'].iloc[-1])
            last_date = df.index[-1]
            ax1.annotate(
                f"{label}: {last_flow:.1f}%",
                xy=(last_date, last_flow),
                xytext=off, textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9),
                fontsize=9, fontweight='bold', color=color,
            )
        except Exception:
            pass

    ax2.plot(rsx1.index, rsx1, label=f"{label1} — RSX(9)", linewidth=1.5, color=color1)
    ax2.plot(rsx2.index, rsx2, label=f"{label2} — RSX(9)", linewidth=1.5, color=color2)
    ax2.axhline(70, color='red',   linestyle='--', linewidth=1)
    ax2.axhline(30, color='green', linestyle='--', linewidth=1)
    ax2.axhline(50, color='gray',  linestyle='-',  alpha=0.4)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSX(9)', fontsize=9)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(alpha=0.3)

    rsx_offsets = [(-70, 25), (-70, -40)]
    for (rsx, color, label), off in zip(((rsx1, color1, label1), (rsx2, color2, label2)), rsx_offsets):
        try:
            last_rsx_date = rsx.index[-1]
            last_rsx_val  = float(rsx.iloc[-1])
            ax2.annotate(
                f"{label} RSX: {last_rsx_val:.1f}",
                xy=(last_rsx_date, last_rsx_val),
                xytext=off, textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9),
                fontsize=9, fontweight='bold', color=color,
            )
        except Exception:
            pass

    draw_perigees(ax1, ax2, perigees, now_ts)
    draw_phase_table(ax3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


# ────────────────────────────────────────────────
# График распределения
# ────────────────────────────────────────────────
def make_distribution_chart():
    assets = list(FUTURES.keys())
    flow_data = {}
    for a in assets:
        df = smart_money_flow(FUTURES[a])
        if df is not None:
            flow_data[a] = df['Flow']
    if not flow_data:
        return None

    fig = plt.figure(figsize=(19, 9))
    gs  = fig.add_gridspec(1, 2, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    assets_list = list(flow_data.keys())
    scores = [float(flow_data[k].iloc[-1]) / 100.0 if len(flow_data[k]) > 0 else 0.0 for k in assets_list]
    bar_colors = [
        '#006400' if s > 0.7 else
        '#32CD32' if s > 0.55 else
        'gray'    if s > 0.45 else
        '#FF8C00' if s > 0.30 else
        '#DC143C'
        for s in scores
    ]
    bars = ax1.bar([a.upper() for a in assets_list], scores, color=bar_colors, edgecolor='black', linewidth=1.0)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Sentiment (0-1)')
    ax1.set_title('Current Sentiment (CFTC, last trading day)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{score*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    colors       = ['#006400', '#32CD32', 'gray', '#FF8C00', '#DC143C']
    level_ranges = [(70, 100), (55, 70), (45, 55), (30, 45), (0, 30)]
    level_names  = ['Strong Bulls', 'Bulls', 'Neutral', 'Bears', 'Strong Bears']
    x     = np.arange(len(assets_list))
    width = 0.15

    for i, (low, high) in enumerate(level_ranges):
        bottom = np.zeros(len(assets_list))
        values = np.array([
            int(((flow_data[asset] > low) & (flow_data[asset] <= high)).sum())
            for asset in assets_list
        ])
        ax2.bar(x + i * width, values, width, bottom=bottom, color=colors[i], edgecolor='black')
        for j, td_count in enumerate(values):
            if td_count > 0:
                ax2.text(x[j] + i * width, bottom[j] + td_count / 2, f'{td_count} d.',
                         ha='center', va='center', fontsize=8, color='black', fontweight='bold')

    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels([a.upper() for a in assets_list], fontsize=11)
    ax2.set_ylabel('Number of trading days')
    ax2.set_title('Distribution over 175 trading days', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 175)
    ax2.grid(axis='y', alpha=0.3)
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[i],
                   markersize=10, label=level_names[i])
        for i in range(len(level_names))
    ]
    ax2.legend(handles=legend_elements, fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.suptitle(
        'Sentiment: ' + ', '.join([a.upper() for a in assets_list]) +
        ' (CFTC, 175 Trading Days)\nby Megatrend',
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


# ────────────────────────────────────────────────
# Хелпер: получение данных с повторными попытками
# ────────────────────────────────────────────────
async def fetch_df_with_retries(ticker: str, label: str, attempts: int = 3, delay: int = 10):
    df = None
    for attempt in range(attempts):
        df = smart_money_flow(ticker)
        if df is not None:
            break
        logger.warning(f"{label}: попытка {attempt+1} вернула None, повтор через {delay} сек...")
        await asyncio.sleep(delay)
    return df


# ────────────────────────────────────────────────
# Ежедневная отправка в 03:00 UTC
# ────────────────────────────────────────────────
async def daily_sender():
    if not CHAT_ID:
        logger.warning("CHAT_ID не задан — ежедневная отправка отключена")
        return

    while True:
        now      = datetime.now(timezone.utc)
        next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if now >= next_run:
            next_run += timedelta(days=1)
        wait_sec = (next_run - now).total_seconds()
        logger.info(f"Ежедневная отправка через {wait_sec:.0f} сек ({next_run.strftime('%Y-%m-%d %H:%M UTC')})")
        await asyncio.sleep(wait_sec)

        logger.info("Запуск ежедневной отправки графиков...")
        try:
            for cmd, ticker in FUTURES.items():
                df = await fetch_df_with_retries(ticker, cmd.upper())

                if df is None:
                    logger.error(f"{cmd.upper()}: все попытки исчерпаны, пропускаем")
                    await bot_app.bot.send_message(
                        chat_id=CHAT_ID,
                        text=f"⚠️ {cmd.upper()}: не удалось получить данные после 3 попыток"
                    )
                    continue

                buf = make_chart(df, cmd.upper())
                await bot_app.bot.send_photo(
                    chat_id=CHAT_ID,
                    photo=buf,
                    caption=f"{cmd.upper()} — Volume Stress / Participation Index + RSX(9)"
                )
                logger.info(f"{cmd.upper()}: отправлен ✅")

            for key, cfg in COMBINED.items():
                df1 = await fetch_df_with_retries(cfg['sym1'], cfg['label1'])
                df2 = await fetch_df_with_retries(cfg['sym2'], cfg['label2'])

                if df1 is None or df2 is None:
                    logger.error(f"{key.upper()}: не удалось получить данные для комбинированного графика")
                    await bot_app.bot.send_message(
                        chat_id=CHAT_ID,
                        text=f"⚠️ {key.upper()}: не удалось получить данные после 3 попыток"
                    )
                    continue

                buf = make_combined_chart(df1, df2, cfg)
                await bot_app.bot.send_photo(
                    chat_id=CHAT_ID,
                    photo=buf,
                    caption=f"{cfg['title']} + RSX(9)"
                )
                logger.info(f"{key.upper()}: отправлен ✅")

            buf_dist = make_distribution_chart()
            if buf_dist:
                await bot_app.bot.send_photo(
                    chat_id=CHAT_ID,
                    photo=buf_dist,
                    caption="Distribution (175 Trading Days)"
                )
            logger.info("Ежедневная отправка завершена")
        except Exception as e:
            logger.error(f"Ошибка ежедневной отправки: {e}")


# ────────────────────────────────────────────────
# Команды
# ────────────────────────────────────────────────
async def handle_asset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        asset = update.message.text.replace('/', '').lower()
        if asset not in FUTURES:
            await update.message.reply_text("Unknown command.")
            return
        await update.message.reply_text(f"Fetching {asset.upper()} data...")
        df = smart_money_flow(FUTURES[asset])
        if df is None:
            await update.message.reply_text("Not enough data.")
            return
        rsx       = calculate_rsx(df['Flow'], length=9)
        last_flow = float(df['Flow'].iloc[-1]) if len(df)  > 0 else None
        last_rsx  = float(rsx.iloc[-1])        if len(rsx) > 0 else None
        buf = make_chart(df, asset.upper())
        await update.message.reply_photo(photo=buf)
        txt  = f"{asset.upper()}:\n"
        txt += f"Participation Index: {last_flow:.1f}%\n" if last_flow is not None else "Participation Index: n/a\n"
        txt += f"RSX(9): {last_rsx:.1f}\n"               if last_rsx  is not None else "RSX(9): n/a\n"
        txt += f"Date: {df.index[-1].strftime('%d.%m.%Y')}"
        await update.message.reply_text(txt)
    except Exception as e:
        logger.error(f"Error in handle_asset: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def handle_combined(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        key = update.message.text.replace('/', '').lower()
        cfg = COMBINED.get(key)
        if cfg is None:
            await update.message.reply_text("Unknown command.")
            return
        await update.message.reply_text(f"Fetching {cfg['label1']} + {cfg['label2']} data...")
        df1 = smart_money_flow(cfg['sym1'])
        df2 = smart_money_flow(cfg['sym2'])
        if df1 is None or df2 is None:
            await update.message.reply_text("Not enough data.")
            return
        rsx1 = calculate_rsx(df1['Flow'], length=9)
        rsx2 = calculate_rsx(df2['Flow'], length=9)
        buf = make_combined_chart(df1, df2, cfg)
        await update.message.reply_photo(photo=buf)
        txt = (
            f"{cfg['label1']} vs {cfg['label2']}:\n"
            f"{cfg['label1']} — Participation Index: {float(df1['Flow'].iloc[-1]):.1f}%   RSX(9): {float(rsx1.iloc[-1]):.1f}\n"
            f"{cfg['label2']} — Participation Index: {float(df2['Flow'].iloc[-1]):.1f}%   RSX(9): {float(rsx2.iloc[-1]):.1f}\n"
            f"Date: {df1.index[-1].strftime('%d.%m.%Y')}"
        )
        await update.message.reply_text(txt)
    except Exception as e:
        logger.error(f"Error in handle_combined: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def distribution(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("Generating distribution chart...")
        buf = make_distribution_chart()
        if buf:
            await update.message.reply_photo(photo=buf, caption="Smart Money Flow Distribution (175 Trading Days)")
        else:
            await update.message.reply_text("Could not generate chart.")
    except Exception as e:
        logger.error(f"Error in distribution: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def all_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("Generating all charts...")
        for cmd in FUTURES.keys():
            df = smart_money_flow(FUTURES[cmd])
            if df is None:
                continue
            buf = make_chart(df, cmd.upper())
            await update.message.reply_photo(photo=buf, caption=f"{cmd.upper()} — Volume Stress / Participation Index + RSX(9)")

        for key, cfg in COMBINED.items():
            df1 = smart_money_flow(cfg['sym1'])
            df2 = smart_money_flow(cfg['sym2'])
            if df1 is None or df2 is None:
                continue
            buf = make_combined_chart(df1, df2, cfg)
            await update.message.reply_photo(photo=buf, caption=f"{cfg['title']} + RSX(9)")

        bufd = make_distribution_chart()
        if bufd:
            await update.message.reply_photo(photo=bufd, caption="Distribution")
    except Exception as e:
        logger.error(f"Error in all_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        txt  = "Volume Stress / Participation Index by Megatrend — commands:\n"
        txt += "/gc /cl /pl /6e /6j /dx — charts\n"
        txt += "/cgc — combined GC + PL\n"
        txt += "/ccl — combined CL + Brent\n"
        txt += "/dist — distribution\n"
        txt += "/all — all charts + distribution\n"
        await update.message.reply_text(txt)
    except Exception as e:
        logger.error(f"Error in start_cmd: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


# ────────────────────────────────────────────────
# Webhook + lifespan
# ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await bot_app.initialize()

    # ✅ исправлено: регистрация хендлеров внутри lifespan — один раз, не на уровне модуля
    for cmd in FUTURES.keys():
        bot_app.add_handler(CommandHandler(cmd, handle_asset))
    for cmd in COMBINED.keys():
        bot_app.add_handler(CommandHandler(cmd, handle_combined))
    bot_app.add_handler(CommandHandler("dist",  distribution))
    bot_app.add_handler(CommandHandler("all",   all_command))
    bot_app.add_handler(CommandHandler("start", start_cmd))

    try:
        await bot_app.bot.set_webhook(f"{URL}/webhook")
        logger.info(f"Webhook set: {URL}/webhook")
    except Exception as e:
        logger.error(f"Webhook error: {e}")
    await bot_app.start()
    task = asyncio.create_task(daily_sender())
    yield
    task.cancel()
    await bot_app.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/webhook")
async def webhook(request: Request):
    try:
        json_update = await request.json()
        logger.info(f"Incoming update: {json_update}")
        update = Update.de_json(json_update, bot_app.bot)
        if update:
            await bot_app.process_update(update)
        return {"ok": True}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test-gc")
async def test_gc():
    df = smart_money_flow(FUTURES['gc'])
    if df is None:
        raise HTTPException(status_code=503, detail="Not enough data")
    buf = make_chart(df, 'GC')
    return StreamingResponse(buf, media_type="image/png")


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "SmartMoney Bot alive!"}

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "OK"}

@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    return {"status": "OK"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
