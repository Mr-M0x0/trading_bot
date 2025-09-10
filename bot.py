import os
import logging
import ccxt
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import asyncio

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bitget-signal-bot")

# ================= ENV VARS =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BITGET_KEY = os.getenv("BITGET_API_KEY", "").strip()
BITGET_SECRET = os.getenv("BITGET_API_SECRET", "").strip()
BITGET_PASSPHRASE = os.getenv("BITGET_API_PASSPHRASE", "").strip()
USER_ID = os.getenv("TELEGRAM_USER_ID", "").strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN missing")

# ================= EXCHANGE =================
exchange = ccxt.bitget({
    "apiKey": BITGET_KEY,
    "secret": BITGET_SECRET,
    "password": BITGET_PASSPHRASE,
    "enableRateLimit": True,
})

# ================= STRATEGY =================
def analyze_pair(symbol="BTC/USDT", timeframe="1h"):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
        df["rsi"] = df["close"].pct_change().rolling(14).mean()  # simple RSI placeholder
        latest = df.iloc[-1]

        verdict = "WATCH"
        reasons = []

        if latest["rsi"] < -0.02:
            verdict = "BUY ✅"
            reasons.append("RSI oversold")
        elif latest["rsi"] > 0.02:
            verdict = "SELL ❌"
            reasons.append("RSI overbought")
        else:
            reasons.append("Neutral RSI")

        return f"""
📊 {symbol} | {timeframe}
Price: {latest['close']}
RSI14: {latest['rsi']:.2f}
Verdict: {verdict}
Reasons: {", ".join(reasons)}
"""
    except Exception as e:
        return f"⚠️ Error analyzing {symbol}: {e}"

# ================= COMMAND HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Welcome! Use /signal BTC/USDT or /top to get market signals.")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        symbol = context.args[0].upper()
    else:
        symbol = "BTC/USDT"
    result = analyze_pair(symbol)
    await update.message.reply_text(result)

async def top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pairs = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","ADA/USDT"]
    msgs = ["🔥 Top Market Scan:"]
    for p in pairs:
        msgs.append(analyze_pair(p))
    await update.message.reply_text("\n".join(msgs))

# ================= HOURLY PUSH =================
async def hourly_push(context: ContextTypes.DEFAULT_TYPE):
    if not USER_ID:
        logger.warning("No TELEGRAM_USER_ID set, skipping push.")
        return
    pairs = ["BTC/USDT","ETH/USDT","SOL/USDT"]
    msgs = ["⏰ Hourly Market Update:"]
    for p in pairs:
        msgs.append(analyze_pair(p))
    await context.bot.send_message(chat_id=USER_ID, text="\n".join(msgs))

# ================= MAIN =================
def build_application():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CommandHandler("top", top))
    return app

async def main():
    app = build_application()
    app.job_queue.run_repeating(hourly_push, interval=3600, first=60)
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
