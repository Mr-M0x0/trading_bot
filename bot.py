# bot.py — Beast Mode Signal Bot (Bitget/ccxt + Telegram)
# Async (python-telegram-bot v20), multi-TF, robust scoring, hourly alerts
# WARNING: Signals are probabilistic. Manage risk. Revoke keys if leaked.

import os
import json
import math
import time
import logging
import asyncio
import traceback
from datetime import datetime
from typing import List, Dict

import ccxt
import pandas as pd
import numpy as np
import ta
import aiofiles

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
)

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("beast-signal-bot")

# ---------------------------
# Configuration / Env
# ---------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
USER_CHAT_ID = os.getenv("TELEGRAM_USER_ID", "").strip()  # your private DM id
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "1h")
WATCHLIST_ENV = os.getenv("WATCHLIST", "BTC/USDT,ETH/USDT,BNB/USDT")
WATCHLIST_FILE = "watchlist.json"
ALERT_FREQ_MIN = int(os.getenv("ALERT_FREQUENCY_MIN", "60"))

BITGET_API_KEY = os.getenv("BITGET_API_KEY")
BITGET_API_SECRET = os.getenv("BITGET_API_SECRET")
BITGET_API_PASSPHRASE = os.getenv("BITGET_API_PASSPHRASE")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

# ---------------------------
# Exchange client (sync ccxt used in threads)
# ---------------------------
exchange_kwargs = {}
if BITGET_API_KEY and BITGET_API_SECRET:
    exchange_kwargs = {
        "apiKey": BITGET_API_KEY,
        "secret": BITGET_API_SECRET,
        "password": BITGET_API_PASSPHRASE,
        "enableRateLimit": True,
    }
else:
    exchange_kwargs = {"enableRateLimit": True}

exchange = ccxt.bitget(exchange_kwargs)

# Concurrency control for API calls
API_SEMAPHORE = asyncio.Semaphore(4)  # limit concurrent fetches

# ---------------------------
# Utilities: watchlist persistence
# ---------------------------
async def load_watchlist() -> List[str]:
    if os.path.exists(WATCHLIST_FILE):
        try:
            async with aiofiles.open(WATCHLIST_FILE, "r") as f:
                text = await f.read()
            data = json.loads(text)
            return data.get("watchlist", [])
        except Exception:
            logger.exception("load_watchlist failed, falling back to env")
    # fallback to env
    return [s.strip().upper() for s in WATCHLIST_ENV.split(",") if s.strip()]

async def save_watchlist(wl: List[str]):
    data = {"watchlist": wl}
    async with aiofiles.open(WATCHLIST_FILE, "w") as f:
        await f.write(json.dumps(data, indent=2))

# ---------------------------
# Helpers: fetch OHLCV safely (run sync ccxt in thread)
# ---------------------------
def normalize_symbol(sym: str) -> str:
    s = sym.strip().upper().replace(" ", "")
    if "/" not in s:
        if s.endswith("USDT"): s = s[:-4] + "/USDT"
        elif s.endswith("USD"): s = s[:-3] + "/USD"
    return s

def fetch_ohlcv_sync(symbol: str, timeframe: str, limit: int = 500):
    # synchronous call (ccxt)
    sym = normalize_symbol(symbol)
    return exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)

async def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500, attempts=2, backoff=1.0):
    async with API_SEMAPHORE:
        for i in range(attempts):
            try:
                df_raw = await asyncio.to_thread(fetch_ohlcv_sync, symbol, timeframe, limit)
                df = pd.DataFrame(df_raw, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                # ensure numeric types
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
            except ccxt.NetworkError as e:
                logger.warning("Network error fetching %s %s: %s (attempt %s)", symbol, timeframe, e, i+1)
                await asyncio.sleep(backoff * (i+1))
            except ccxt.ExchangeError as e:
                logger.error("Exchange error fetching %s %s: %s", symbol, timeframe, e)
                raise
            except Exception as e:
                logger.exception("Unexpected fetch error for %s %s: %s", symbol, timeframe, e)
                await asyncio.sleep(backoff * (i+1))
        raise RuntimeError(f"Failed to fetch OHLCV for {symbol} {timeframe}")

# ---------------------------
# Indicators & Scoring (sync code, run in thread)
# ---------------------------
def compute_indicators_sync(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low  = df["low"]

    # EMA
    df["EMA20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()

    # RSI (true RSI)
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
    df["BB_WIDTH"] = (df["BB_UP"] - df["BB_LOW"]) / (df["BB_MID"].replace(0, np.nan))

    # ATR
    df["ATR14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # Volume boost
    df["VOL_MEDIAN30"] = df["volume"].rolling(30).median().replace(0, np.nan)
    df["VOL_BOOST"] = df["volume"] / (df["VOL_MEDIAN30"] + 1e-9)

    return df

async def compute_indicators(df: pd.DataFrame):
    return await asyncio.to_thread(compute_indicators_sync, df)

def score_single_tf(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    # trend
    cond_trend_buy = last["EMA20"] > last["EMA50"]
    cond_trend_sell = last["EMA20"] < last["EMA50"]

    # RSI
    cond_rsi_buy = last["RSI14"] < 40 and last["RSI14"] > prev["RSI14"]
    cond_rsi_sell = last["RSI14"] > 60 and last["RSI14"] < prev["RSI14"]

    # MACD
    cond_macd_buy = last["MACD"] > last["MACD_SIGNAL"] and last["MACD_HIST"] > prev["MACD_HIST"]
    cond_macd_sell = last["MACD"] < last["MACD_SIGNAL"] and last["MACD_HIST"] < prev["MACD_HIST"]

    # squeeze
    recent_bw = df["BB_WIDTH"].tail(40).dropna()
    squeeze = False
    if len(recent_bw) >= 10:
        squeeze = recent_bw.mean() < df["BB_WIDTH"].quantile(0.4)

    breakout_up = last["close"] > last["BB_MID"]
    breakout_down = last["close"] < last["BB_MID"]

    vol_boost = float(last.get("VOL_BOOST", 1.0) or 1.0)
    vol_factor = 1.0
    if vol_boost > 1.2:
        vol_factor = min(1.2, 1.0 + (vol_boost - 1.0) * 0.15)

    buy_score = 0.0
    sell_score = 0.0
    if cond_trend_buy: buy_score += 0.30
    if cond_rsi_buy:   buy_score += 0.25
    if cond_macd_buy:  buy_score += 0.25
    if squeeze and breakout_up: buy_score += 0.20
    buy_score *= vol_factor

    if cond_trend_sell: sell_score += 0.30
    if cond_rsi_sell:   sell_score += 0.25
    if cond_macd_sell:  sell_score += 0.25
    if squeeze and breakout_down: sell_score += 0.20
    sell_score *= vol_factor

    verdict = "WATCH"
    if buy_score >= 0.6 and buy_score > sell_score: verdict = "BUY"
    elif sell_score >= 0.6 and sell_score > buy_score: verdict = "SELL"

    atr = float(last.get("ATR14", 0.0) or 0.0)
    price = float(last["close"])
    sl = tp = None
    if verdict == "BUY" and atr > 0:
        sl = round(max(last["BB_LOW"], price - 1.5 * atr), 8)
        tp = round(price + 2.5 * atr, 8)
    elif verdict == "SELL" and atr > 0:
        sl = round(min(last["BB_UP"], price + 1.5 * atr), 8)
        tp = round(price - 2.5 * atr, 8)

    reasons = []
    if cond_trend_buy: reasons.append("EMA trend bullish")
    if cond_rsi_buy:   reasons.append("RSI recovering")
    if cond_macd_buy:  reasons.append("MACD bullish")
    if squeeze and breakout_up: reasons.append("Squeeze→BreakoutUp")
    if cond_trend_sell: reasons.append("EMA trend bearish")
    if cond_rsi_sell:   reasons.append("RSI cooling")
    if cond_macd_sell:  reasons.append("MACD bearish")
    if squeeze and breakout_down: reasons.append("Squeeze→Breakdown")
    if vol_boost > 1.2: reasons.append("Volume surge")

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
        "reasons": reasons,
    }

async def analyze_multi(symbol: str, tfs=("1h","4h","1d")) -> Dict:
    symbol = normalize_symbol(symbol)
    per = {}
    buy_votes = sell_votes = 0
    sum_buy = sum_sell = 0.0
    price = None
    for tf in tfs:
        df = await fetch_ohlcv(symbol, tf, limit=500 if tf!="1d" else 300)
        df = await compute_indicators(df)
        res = await asyncio.to_thread(score_single_tf, df)
        per[tf] = res
        if price is None:
            price = res["price"]
        sum_buy += res["buy_score"]
        sum_sell += res["sell_score"]
        if res["verdict"] == "BUY": buy_votes += 1
        if res["verdict"] == "SELL": sell_votes += 1
        await asyncio.sleep(0.05)
    avg_buy = round(sum_buy / len(tfs), 2)
    avg_sell = round(sum_sell / len(tfs), 2)

    final = "WATCH"
    sl = tp = None
    # strict confirmation: majority + avg threshold
    if buy_votes >= 2 and avg_buy >= 0.65 and avg_buy > avg_sell:
        final = "BUY"
        sl = per[tfs[0]]["stop_loss"]; tp = per[tfs[0]]["target"]
    elif sell_votes >= 2 and avg_sell >= 0.65 and avg_sell > avg_buy:
        final = "SELL"
        sl = per[tfs[0]]["stop_loss"]; tp = per[tfs[0]]["target"]

    return {
        "symbol": symbol,
        "price": price,
        "avg_buy": avg_buy,
        "avg_sell": avg_sell,
        "verdict": final,
        "sl": sl, "tp": tp,
        "per_tf": per,
    }

# ---------------------------
# Telegram UI helpers
# ---------------------------
def pair_keyboard(pairs: List[str], cols=3):
    buttons = []
    row = []
    for i, p in enumerate(pairs, 1):
        row.append(InlineKeyboardButton(p, callback_data=f"SIG|{p}"))
        if i % cols == 0:
            buttons.append(row); row = []
    if row: buttons.append(row)
    return InlineKeyboardMarkup(buttons)

# ---------------------------
# Handlers: commands
# ---------------------------
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "🤖 Beast Signal Bot — Multi-TF, volume aware\n\n"
        "Commands:\n"
        "/signal <PAIR> — analyze a symbol (e.g. /signal BTC/USDT)\n"
        "/signal — show quick pairs\n"
        "/scan — rank your watchlist\n"
        "/watchlist — show current list\n"
        "/add <PAIR> — add to watchlist\n"
        "/remove <PAIR> — remove from watchlist\n"
        "/forceupdate — run full scan now\n"
    )
    await update.message.reply_text(txt)

async def signal_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        wl = await load_watchlist()
        await update.message.reply_text("Pick a pair:", reply_markup=pair_keyboard(wl[:12]))
        return
    symbol = args[0].upper()
    await update.message.reply_text(f"Analyzing {symbol} (multi-TF)... this can take a few seconds.")
    try:
        res = await analyze_multi(symbol, tfs=("1h","4h","1d"))
        lines = [f"📊 {res['symbol']}  (multi-TF 1h/4h/1d)"]
        lines.append(f"Price: {res['price']:.8f}")
        lines.append(f"Avg buy/sell scores: B{res['avg_buy']} / S{res['avg_sell']}")
        lines.append(f"Verdict: {res['verdict']}")
        if res["sl"] and res["tp"]:
            lines.append(f"SL: {res['sl']}  TP: {res['tp']}")
        # per-tf condensed:
        details = []
        for tf, data in res["per_tf"].items():
            details.append(f"{tf}:{data['verdict']} (B{data['buy_score']}/S{data['sell_score']}) RSI{data['rsi']:.1f}")
        lines.append("TF → " + " • ".join(details))
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        logger.exception("signal_handler failed")
        await update.message.reply_text(f"Error analyzing {symbol}: {e}")

async def callback_sig(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data
    if data.startswith("SIG|"):
        pair = data.split("|",1)[1]
        await q.edit_message_text("Analyzing " + pair + " ...")
        try:
            res = await analyze_multi(pair, tfs=("1h","4h","1d"))
            lines = [f"📊 {res['symbol']}  (multi-TF 1h/4h/1d)"]
            lines.append(f"Price: {res['price']:.8f}")
            lines.append(f"Avg buy/sell scores: B{res['avg_buy']} / S{res['avg_sell']}")
            lines.append(f"Verdict: {res['verdict']}")
            if res["sl"] and res["tp"]:
                lines.append(f"SL: {res['sl']}  TP: {res['tp']}")
            details = []
            for tf, data in res["per_tf"].items():
                details.append(f"{tf}:{data['verdict']} (B{data['buy_score']}/S{data['sell_score']}) RSI{data['rsi']:.1f}")
            lines.append("TF → " + " • ".join(details))
            await q.edit_message_text("\n".join(lines))
        except Exception as e:
            logger.exception("callback_sig failed")
            await q.edit_message_text(f"Error: {e}")

async def scan_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Scanning watchlist (this may take 20-60s)...")
    wl = await load_watchlist()
    results = []
    for p in wl:
        try:
            r = await analyze_multi(p, tfs=("1h","4h","1d"))
            results.append(r)
        except Exception as e:
            logger.warning("scan: failed %s -> %s", p, e)
    if not results:
        await update.message.reply_text("No results.")
        return
    ranked = sorted(results, key=lambda x: (x["verdict"]=="BUY", x["avg_buy"], -x["avg_sell"]), reverse=True)
    lines = ["🔎 Watchlist Scan Top (MTF):"]
    for r in ranked[:12]:
        s = f"{r['symbol']} → {r['verdict']} | B{r['avg_buy']} S{r['avg_sell']} | Price {r['price']:.6f}"
        if r["sl"] and r["tp"]:
            s += f" | SL {r['sl']} TP {r['tp']}"
        lines.append(s)
    await update.message.reply_text("\n".join(lines))

async def watchlist_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    wl = await load_watchlist()
    await update.message.reply_text("Watchlist:\n" + ", ".join(wl))

async def add_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /add SYMBOL (e.g. /add SOL/USDT)")
        return
    sym = normalize_symbol(args[0])
    wl = await load_watchlist()
    if sym in wl:
        await update.message.reply_text(f"{sym} already in watchlist.")
        return
    wl.insert(0, sym)
    await save_watchlist(wl)
    await update.message.reply_text(f"Added {sym} to watchlist.")

async def remove_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /remove SYMBOL")
        return
    sym = normalize_symbol(args[0])
    wl = await load_watchlist()
    if sym not in wl:
        await update.message.reply_text(f"{sym} not found.")
        return
    wl = [x for x in wl if x != sym]
    await save_watchlist(wl)
    await update.message.reply_text(f"Removed {sym} from watchlist.")

async def forceupdate_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Manual full scan started (may take up to 1-2 minutes)...")
    # run same as hourly push but reply the results to user
    wl = await load_watchlist()
    results = []
    for p in wl[:20]:
        try:
            r = await analyze_multi(p, tfs=("1h","4h","1d"))
            if r["verdict"] == "BUY" or r["avg_buy"] >= 0.7:
                results.append(r)
        except Exception as e:
            logger.warning("forceupdate failed %s: %s", p, e)
    if not results:
        await update.message.reply_text("No high-conviction setups found.")
    else:
        lines = ["🚨 High-conviction setups:"]
        for r in sorted(results, key=lambda x: x["avg_buy"], reverse=True)[:8]:
            s = f"{r['symbol']} → {r['verdict']} | B{r['avg_buy']} S{r['avg_sell']} | {r['price']:.6f}"
            if r["sl"] and r["tp"]:
                s += f" | SL {r['sl']} TP {r['tp']}"
            lines.append(s)
        await update.message.reply_text("\n".join(lines))

# ---------------------------
# Hourly push job
# ---------------------------
async def hourly_job(context: ContextTypes.DEFAULT_TYPE):
    if not USER_CHAT_ID:
        logger.info("USER_CHAT_ID not set; skipping hourly push.")
        return
    wl = await load_watchlist()
    results = []
    for p in wl[:40]:
        try:
            r = await analyze_multi(p, tfs=("1h","4h","1d"))
            if r["verdict"] == "BUY" and r["avg_buy"] >= 0.7:
                results.append(r)
        except Exception as e:
            logger.warning("hourly job fail %s: %s", p, e)
    if not results:
        await context.bot.send_message(chat_id=int(USER_CHAT_ID), text="⏰ Hourly scan: no high-conviction setups right now.")
        return
    top = sorted(results, key=lambda x: x["avg_buy"], reverse=True)[:6]
    lines = ["⏰ Hourly high-conviction setups:"]
    for r in top:
        s = f"{r['symbol']} → BUY | B{r['avg_buy']} S{r['avg_sell']} | {r['price']:.6f}"
        if r["sl"] and r["tp"]:
            s += f" | SL {r['sl']} TP {r['tp']}"
        lines.append(s)
    await context.bot.send_message(chat_id=int(USER_CHAT_ID), text="\n".join(lines))

# ---------------------------
# App build & run
# ---------------------------
def build_app():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("signal", signal_handler))
    app.add_handler(CallbackQueryHandler(callback_sig, pattern="^SIG\\|"))
    app.add_handler(CommandHandler("scan", scan_handler))
    app.add_handler(CommandHandler("watchlist", watchlist_handler))
    app.add_handler(CommandHandler("add", add_handler))
    app.add_handler(CommandHandler("remove", remove_handler))
    app.add_handler(CommandHandler("forceupdate", forceupdate_handler))
    return app

async def main():
    # ensure watchlist file exists
    if not os.path.exists(WATCHLIST_FILE):
        wl = [s.strip().upper() for s in WATCHLIST_ENV.split(",") if s.strip()]
        await save_watchlist(wl)

    app = build_app()
    # schedule hourly job
    try:
        app.job_queue.run_repeating(hourly_job, interval=ALERT_FREQ_MIN*60, first=30)
    except Exception as e:
        logger.warning("Failed to schedule hourly job: %s", e)

    logger.info("Starting bot (polling)...")
    # delete webhook if any (sync)
    try:
        await asyncio.to_thread(lambda: exchange)  # noop to warm exchange
    except Exception:
        pass
    await app.run_polling()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        logger.exception("Fatal error in main")
