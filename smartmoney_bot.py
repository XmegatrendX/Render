import io
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from contextlib import asynccontextmanager
import logging
import asyncio
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("BOT_TOKEN")
URL = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME', 'smartmoney-bot.onrender.com')}"

async def keep_alive():
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://www.google.com") as resp:
                    logger.info(f"Keep-alive ping: {resp.status}")
        except Exception as e:
            logger.error(f"Keep-alive error: {e}")
        await asyncio.sleep(300)

application = Application.builder().token(TOKEN).build()

FUTURES = {
    'gc': 'GC=F', 'cl': 'CL=F', 'pl': 'PL=F',
    '6e': '6E=F', '6j': '6J=F', 'dx': 'DX=F'
}

# === ВСЕ ТЕ ЖЕ ФУНКЦИИ smart_money_flow, calculate_rsx, make_chart, make_distribution_chart ===
# (скопируй их из моего предыдущего сообщения — они не меняются)

def smart_money_flow(symbol, days=175):
    df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)
    if df is None or len(df) < 20: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
    df['Price_Acc'] = df['Close'].pct_change().diff().fillna(0)
    df['Signal'] = 0.8 * df['Vol_Z'] + 0.2 * df['Price_Acc']
    df['Flow'] = (df['Signal'].clip(-3, 3) * 16.67 + 50).ewm(span=3).mean()
    return df

def calculate_rsx(close, period=9):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period).mean()
    down = (-delta).clip(lower=0).ewm(alpha=1/period).mean()
    rs = up / (down + 1e-8)
    return 100 - (100 / (1 + rs))

def make_chart(df, symbol):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['Flow'], label="Smart Money Flow", linewidth=2)
    ax.axhline(85, color='red', linestyle='--', label='Sell Zone')
    ax.axhline(15, color='green', linestyle='--', label='Buy Zone')
    ax.axhline(50, color='gray', linestyle='-', alpha=0.5)
    ax.set_title(f"{symbol} — Smart Money Flow by Megatrend", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)
    try:
        last_flow = float(df['Flow'].iloc[-1])
        last_date = df.index[-1]
        ax.text(last_date, last_flow, f"{last_flow:.1f}%", fontsize=10, fontweight='bold',
                ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    except: pass
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def make_distribution_chart():
    assets = list(FUTURES.keys())
    flow_data = {}
    for a in assets:
        df = smart_money_flow(FUTURES[a])
        if df is not None:
            flow_data[a] = df['Flow']
    if not flow_data: return None
    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    assets_list = list(flow_data.keys())
    scores = [float(flow_data[k].iloc[-1]) / 100.0 if len(flow_data[k]) > 0 else 0.0 for k in assets_list]
    bar_colors = ['#006400' if s > 0.7 else '#32CD32' if s > 0.55 else 'gray' if s > 0.45 else '#FF8C00' if s > 0.30 else '#DC143C' for s in scores]
    bars = ax1.bar([a.upper() for a in assets_list], scores, color=bar_colors, edgecolor='black', linewidth=1.0)
    ax1.set_ylim(0, 1)
    ax1.set_title('Current Sentiment', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{score*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['#006400', '#32CD32', 'gray', '#FF8C00', '#DC143C']
    level_ranges = [(70, 100), (55, 70), (45, 55), (30, 45), (0, 30)]
    x = np.arange(len(assets_list))
    width = 0.15
    bottom = np.zeros(len(assets_list))
    for i, (low, high) in enumerate(level_ranges):
        values = [((flow_data[asset] > low) & (flow_data[asset] <= high)).sum() for asset in assets_list]
        ax2.bar(x + i*width, values, width, bottom=bottom, color=colors[i], edgecolor='black')
        bottom += np.array(values)
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels([a.upper() for a in assets_list], fontsize=11)
    ax2.set_title('Distribution over 175 days', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 175)
    ax2.grid(axis='y', alpha=0.3)
    plt.suptitle('Sentiment: ' + ', '.join([a.upper() for a in assets_list]) + ' by Megatrend', fontsize=14, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# === КОМАНДЫ ===
async def handle_asset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    asset = update.message.text.replace('/', '').lower()
    if asset not in FUTURES:
        await update.message.reply_text("Unknown command.")
        return
    await update.message.reply_text(f"Fetching {asset.upper()} data...")
    df = smart_money_flow(FUTURES[asset])
    if df is None:
        await update.message.reply_text("Not enough data.")
        return
    rsx = calculate_rsx(df['Close'])
    last_flow = float(df['Flow'].iloc[-1])
    last_rsx = float(rsx.iloc[-1])
    buf = make_chart(df, asset.upper())
    await update.message.reply_photo(photo=buf)
    txt = f"{asset.upper()}:\nSmart Money Flow: {last_flow:.1f}%\nRSX(9): {last_rsx:.1f}\nDate: {df.index[-1].strftime('%d.%m.%Y')}"
    await update.message.reply_text(txt)

async def distribution(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating distribution chart...")
    buf = make_distribution_chart()
    if buf:
        await update.message.reply_photo(photo=buf, caption="Smart Money Flow Distribution (175 Days)")
    else:
        await update.message.reply_text("Could not generate chart.")

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = "Smart Money Flow by Megatrend — commands:\n/gc /cl /pl /6e /6j /dx — charts\n/dist — distribution"
    await update.message.reply_text(txt)

for cmd in FUTURES.keys():
    application.add_handler(CommandHandler(cmd, handle_asset))
application.add_handler(CommandHandler("dist", distribution))
application.add_handler(CommandHandler("start", start_cmd))

@asynccontextmanager
async def lifespan(app: FastAPI):
    await application.initialize()
    await application.start()
    asyncio.create_task(keep_alive())
    await application.bot.set_webhook(f"{URL}/webhook")
    logger.info(f"Webhook set: {URL}/webhook")
    yield
    await application.stop()
    await application.shutdown()

app = FastAPI(lifespan=lifespan)

@app.post("/webhook")
async def webhook(request: Request):
    json_update = await request.json()
    update = Update.de_json(json_update, application.bot)
    await application.process_update(update)
    return {"ok": True}

@app.get("/")
async def root():
    return {"status": "SmartMoney Bot alive on Render!"}
