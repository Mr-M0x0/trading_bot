# bot.py
import os
import math
import time
import logging
import asyncio
import ccxt
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# indicator library
import ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bitget-signal-bot")

# ---- Config from env ----
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "15m")
WATCHLIST = [s.strip().upper() for s in os.getenv(
    "WATCHLIST",
    "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT,MATIC/USDT"
).split(",") if s.strip()]

# optional Bitget keys (for future trading or private endpoints)
BITGET_API_KEY = os.getenv("BITGET_API_KEY")
BITGET_API_SECRET = os.getenv("BITGET_API_SECRET")
BITGET_API_PASSPHRASE = os.getenv("BITGET_API_PASSPHRASE")

# ---- CCXT exchange client (public data) ----
exchange_kwargs = {}
if BITGET_API_KEY and BITGET_API_SECRET:
    exchange_kwargs = {
        "apiKey": BITGET_API_KEY,
        "secret": BITGET_API_SECRET,
    }
exchange = ccxt.bitget(exchange_kwargs)

# Mapping user-friendly intervals to ccxt timeframes
VALID_TIMEFRAMES = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h",
    "6h": "6h", "12h": "12h", "1d": "1d"
}

# ---- Helper: fetch OHLCV into DataFrame ----
def fetch_ohlcv_df(symbol: str, timeframe: str = "15m", limit: int = 500) -> pd.DataFrame:
    tf = VALID_TIMEFRAMES.get(timeframe, timeframe)
    # ccxt uses format like 'BTC/USDT'
    if "/" not in symbol:
        # try to correct common formats
        if symbol.endswith("USDT"):
            symbol = symbol.replace("USDT", "/USDT")
        elif symbol.endswith("USD"):
            symbol = symbol.replace("USD", "/USD")
    # fetch
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

# ---- Indicators & scoring function ----
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    df = df.copy()
    # EMA
    df["EMA20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    # RSI
    df["RSI14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    # MACD
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_HIST"] = macd.macd_diff()
    # Bollinger
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_UP"] = bb.bollinger_hband()
    df["BB_MID"] = bb.bollinger_mavg()
    df["BB_LOW"] = bb.bollinger_lband()
    df["BB_WIDTH"] = (df["BB_UP"] - df["BB_LOW"]) / df["BB_MID"]
    # ATR
    df["ATR14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], close, window=14).average_true_range()
    return df

def score_and_recommend(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    # conditions
    cond_trend_buy = (last["EMA20"] > last["EMA50"]) or (prev["EMA20"] <= prev["EMA50"] and last["EMA20"] > last["EMA50"])
    cond_trend_sell = (last["EMA20"] < last["EMA50"]) or (prev["EMA20"] >= prev["EMA50"] and last["EMA20"] < last["EMA50"])
    cond_rsi_buy = last["RSI14"] < 35 and last["RSI14"] > prev["RSI14"]
    cond_rsi_sell = last["RSI14"] > 65 and last["RSI14"] < prev["RSI14"]
    cond_macd_buy = last["MACD"] > last["MACD_SIGNAL"] and last["MACD_HIST"] > prev["MACD_HIST"]
    cond_macd_sell = last["MACD"] < last["MACD_SIGNAL"] and last["MACD_HIST"] < prev["MACD_HIST"]
    squeeze = df["BB_WIDTH"].tail(20).mean() < df["BB_WIDTH"].quantile(0.35)
    breakout_up = last["close"] > last["BB_MID"]
    breakout_down = last["close"] < last["BB_MID"]

    # weighted scoring (tweakable)
    buy_score = 0.0
    sell_score = 0.0
    if cond_trend_buy: buy_score += 0.30
    if cond_rsi_buy: buy_score += 0.25
    if cond_macd_buy: buy_score += 0.25
    if squeeze and breakout_up: buy_score += 0.20

    if cond_trend_sell: sell_score += 0.30
    if cond_rsi_sell: sell_score += 0.25
    if cond_macd_sell: sell_score += 0.25
    if squeeze and breakout_down: sell_score += 0.20

    # Verdict rules
    verdict = "WATCH"
    if buy_score >= 0.6 and buy_score > sell_score:
        verdict = "BUY"
    elif sell_score >= 0.6 and sell_score > buy_score:
        verdict = "SELL"

    # SL / TP using ATR
    atr = float(last.get("ATR14", 0.0) or 0.0)
    price = float(last["close"])
    sl = None
    tp = None
    if verdict == "BUY" and atr > 0:
        sl = round(max(last["BB_LOW"], price - 1.5 * atr), 8)
        tp = round(price + 2.5 * atr, 8)
    elif verdict == "SELL" and atr > 0:
        sl = round(min(last["BB_UP"], price + 1.5 * atr), 8)
        tp = round(price - 2.5 * atr, 8)

    # human-readable reasoning
    reasons = []
    if cond_trend_buy: reasons.append("EMA trend bullish")
    if cond_rsi_buy: reasons.append("RSI recovering from oversold")
    if cond_macd_buy: reasons.append("MACD bullish momentum")
    if squeeze and breakout_up: reasons.append("Bollinger squeeze -> breakout up")
    if cond_trend_sell: reasons.append("EMA trend bearish")
    if cond_rsi_sell: reasons.append("RSI dropping from overbought")
    if cond_macd_sell: reasons.append("MACD bearish momentum")
    if squeeze and breakout_down: reasons.append("Bollinger squeeze -> breakout down")

    return {
        "price": price,
        "rsi": float(last["RSI14"]),
        "ema20": float(last["EMA20"]),
        "ema50": float(last["EMA50"]),
        "macd": float(last["MACD"]),
        "macd_signal": float(last["MACD_SIGNAL"]),
        "bb_width": float(last["BB_WIDTH"]) if not math.isnan(last["BB_WIDTH"]) else None,
        "atr": atr,
        "buy_score": round(buy_score, 2),
        "sell_score": round(sell_score, 2),
        "verdict": verdict,
        "stop_loss": sl,
        "target": tp,
        "reasons": reasons
    }

# ---- Telegram command handlers ----
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "🤖 Bitget Signal Bot\n\n"
        "Commands:\n"
        "/signal SYMBOL [interval]  — get analysis for a pair (e.g. /signal BTC/USDT 1h)\n"
        "/scan [interval]          — scan watchlist and show top opportunities\n"
        "/watchlist                — show current watchlist\n"
        "/config                   — show bot settings\n\n"
        "Signals combine RSI + EMA trend + MACD + Bollinger squeeze with ATR-based SL/TP.\n"
        "I only give signals (no auto-trading). Not financial advice."
    )
    await update.message.reply_text(txt)

async def config_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = f"Default interval: {DEFAULT_INTERVAL}\nWatchlist size: {len(WATCHLIST)}"
    await update.message.reply_text(msg)

async def watchlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Watchlist:\n" + ", ".join(WATCHLIST))

async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /signal SYMBOL [interval] e.g. /signal BTC/USDT 1h")
        return
    symbol = context.args[0].upper()
    interval = context.args[1] if len(context.args) > 1 else DEFAULT_INTERVAL
    try:
        df = fetch_ohlcv_df(symbol, interval, limit=500)
        df = compute_indicators(df)
        res = score_and_recommend(df)
        text = (
            f"📊 {symbol} | {interval}\n"
            f"Price: {res['price']:.8f}\n"
            f"RSI14: {res['rsi']:.2f}\n"
            f"EMA20/50: {res['ema20']:.6f} / {res['ema50']:.6f}\n"
            f"MACD: {res['macd']:.6f} (sig {res['macd_signal']:.6f})\n"
            f"Scores → Buy: {res['buy_score']} | Sell: {res['sell_score']}\n"
            f"Verdict: {res['verdict']}\n"
        )
        if res['stop_loss'] and res['target']:
            text += f"SL: {res['stop_loss']}  TP: {res['target']}\n"
        if res['reasons']:
            text += "Reasons: " + "; ".join(res['reasons']) + "\n"
        await update.message.reply_text(text)
    except Exception as e:
        logger.error("signal error: %s", traceback.format_exc())
        await update.message.reply_text(f"Error analyzing {symbol}: {e}")

async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    interval = context.args[0] if context.args else DEFAULT_INTERVAL
    results = []
    for raw in WATCHLIST:
        sym = raw.replace(" ", "")
        try:
            df = fetch_ohlcv_df(sym, interval, limit=300)
            df = compute_indicators(df)
            res = score_and_recommend(df)
            results.append(res | {"symbol": sym})
            await asyncio.sleep(0.15)  # gentle on API
        except Exception as e:
            logger.warning("scan: failed %s -> %s", sym, e)
    # rank: prefer BUY with higher buy_score then lower sell_score
    ranked = sorted(results, key=lambda r: (r["verdict"] == "BUY", r["buy_score"], -r["sell_score"]), reverse=True)
    top = ranked[:8]
    lines = [f"🔎 Top setups ({interval})"]
    for r in top:
        line = f"{r['symbol']} → {r['verdict']} | Buy {r['buy_score']} Sell {r['sell_score']} | Price {r['price']:.6f}"
        if r["stop_loss"] and r["target"]:
            line += f" | SL {r['stop_loss']} TP {r['target']}"
        lines.append(line)
    if not top:
        lines = ["No strong signals right now."]
    await update.message.reply_text("\n".join(lines))

# ---- Main ----
def build_and_run():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN missing")
        return
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", start_cmd))
    app.add_handler(CommandHandler("config", config_cmd))
    app.add_handler(CommandHandler("watchlist", watchlist_cmd))
    app.add_handler(CommandHandler("signal", signal_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))

    logger.info("Starting Telegram bot...")
    app.run_polling()

if __name__ == "__main__":
    build_and_run()
