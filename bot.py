import os
import logging
import ta
import pandas as pd
import requests
from binance.client import Client
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Binance client
binance_client = Client()

# Env variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "15m")
WATCHLIST = os.getenv("WATCHLIST", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")

# Fetch OHLCV data
def get_klines(symbol, interval="15m", limit=100):
    try:
        klines = binance_client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp','o','h','l','c','v','close_time',
            'qav','num_trades','taker_base_vol','taker_quote_vol','ignore'
        ])
        df['c'] = df['c'].astype(float)
        df['h'] = df['h'].astype(float)
        df['l'] = df['l'].astype(float)
        df['o'] = df['o'].astype(float)
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

# Strategy logic
def analyze(df):
    signals = []
    if df is None or len(df) < 50:
        return "Not enough data."

    close = df['c']

    # Indicators
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    ema50 = ta.trend.EMAIndicator(close, 50).ema_indicator().iloc[-1]
    ema200 = ta.trend.EMAIndicator(close, 200).ema_indicator().iloc[-1]
    macd = ta.trend.MACD(close)
    macd_val = macd.macd().iloc[-1]
    macd_signal = macd.macd_signal().iloc[-1]
    boll = ta.volatility.BollingerBands(close)
    bandwidth = boll.bollinger_hband().iloc[-1] - boll.bollinger_lband().iloc[-1]
    atr = ta.volatility.AverageTrueRange(
        df['h'], df['l'], close, window=14
    ).average_true_range().iloc[-1]

    price = close.iloc[-1]

    # Scoring
    score = 0
    if rsi < 30: score += 1; signals.append("RSI: Oversold ✅")
    if rsi > 70: score -= 1; signals.append("RSI: Overbought ❌")
    if ema50 > ema200: score += 1; signals.append("Trend: Bullish ✅")
    if ema50 < ema200: score -= 1; signals.append("Trend: Bearish ❌")
    if macd_val > macd_signal: score += 1; signals.append("MACD: Bullish ✅")
    if macd_val < macd_signal: score -= 1; signals.append("MACD: Bearish ❌")
    if bandwidth < (0.05 * price): signals.append("⚠️ Bollinger squeeze (possible breakout)")

    stop_loss = round(price - 1.5 * atr, 2)
    take_profit = round(price + 3 * atr, 2)

    if score >= 2:
        final = f"✅ BUY signal\nStop Loss: {stop_loss}\nTake Profit: {take_profit}"
    elif score <= -2:
        final = f"❌ SELL signal\nStop Loss: {take_profit}\nTake Profit: {stop_loss}"
    else:
        final = "⚖ Neutral / Wait"

    return f"Price: {price}\nScore: {score}\n" + "\n".join(signals) + "\n\n" + final

# Telegram commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome! Use /signal BTCUSDT 1h or /scan 15m.")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /signal SYMBOL [interval]")
        return
    symbol = context.args[0].upper()
    interval = context.args[1] if len(context.args) > 1 else DEFAULT_INTERVAL
    df = get_klines(symbol, interval)
    result = analyze(df)
    await update.message.reply_text(f"📊 {symbol} ({interval})\n\n{result}")

async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    interval = context.args[0] if context.args else DEFAULT_INTERVAL
    results = []
    for symbol in WATCHLIST:
        df = get_klines(symbol, interval)
        res = analyze(df)
        if "BUY" in res:
            results.append(f"✅ {symbol}")
        elif "SELL" in res:
            results.append(f"❌ {symbol}")
    if not results:
        results = ["⚖ No strong signals."]
    await update.message.reply_text("\n".join(results))

# Run bot
if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CommandHandler("scan", scan))
    app.run_polling()
