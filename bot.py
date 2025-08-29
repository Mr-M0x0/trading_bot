# webhook cleanup — run the async delete_webhook properly
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
import os, logging, asyncio, math, traceback
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
)
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bitget-signal-bot")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

# Delete any webhook so polling won’t conflict
try:
    res = asyncio.run(Bot(TELEGRAM_TOKEN).delete_webhook())
    logging.info("delete_webhook result: %s", res)
    print("Deleted existing webhook (if any). Result:", res)
except Exception as e:
    logging.warning("Webhook deletion failed (ok to ignore): %s", e)

# ---- Config ----
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "1h")
WATCHLIST = [s.strip().upper() for s in os.getenv(
    "WATCHLIST", "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT,MATIC/USDT"
).split(",") if s.strip()]
ALERT_FREQUENCY_MIN = int(os.getenv("ALERT_FREQUENCY_MIN", "60"))
USER_CHAT_ID = os.getenv("USER_CHAT_ID", "").strip()  # your DM

BITGET_API_KEY = os.getenv("BITGET_API_KEY")
BITGET_API_SECRET = os.getenv("BITGET_API_SECRET")
BITGET_API_PASSPHRASE = os.getenv("BITGET_API_PASSPHRASE")

exchange_kwargs = {}
if BITGET_API_KEY and BITGET_API_SECRET:
    exchange_kwargs = { "apiKey": BITGET_API_KEY, "secret": BITGET_API_SECRET }
exchange = ccxt.bitget(exchange_kwargs)

VALID_TIMEFRAMES = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m","1h":"1h",
    "2h":"2h","4h":"4h","6h":"6h","12h":"12h","1d":"1d"
}

# ----------- Data helpers -----------
def normalize_symbol(symbol: str) -> str:
    s = symbol.upper().replace(" ", "")
    if "/" not in s:
        if s.endswith("USDT"): s = s[:-4] + "/USDT"
        elif s.endswith("USD"): s = s[:-3] + "/USD"
    return s

def fetch_ohlcv_df(symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
    tf = VALID_TIMEFRAMES.get(timeframe, timeframe)
    sym = normalize_symbol(symbol)
    ohlcv = exchange.fetch_ohlcv(sym, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).tz_convert("UTC")
    df.set_index("timestamp", inplace=True)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low  = df["low"]

    # Trend
    df["EMA20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()

    # Momentum
    df["RSI14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_HIST"] = macd.macd_diff()

    # Volatility / structure
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_UP"] = bb.bollinger_hband()
    df["BB_MID"] = bb.bollinger_mavg()
    df["BB_LOW"] = bb.bollinger_lband()
    df["BB_WIDTH"] = (df["BB_UP"] - df["BB_LOW"]) / df["BB_MID"]

    # ATR
    df["ATR14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # Volume filter (relative to recent median)
    vol = df["volume"].astype(float)
    df["VOL_BOOST"] = vol / (vol.rolling(30).median() + 1e-9)

    return df

def score_single_timeframe(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    cond_trend_buy  = (last["EMA20"] > last["EMA50"]) or (prev["EMA20"] <= prev["EMA50"] and last["EMA20"] > last["EMA50"])
    cond_trend_sell = (last["EMA20"] < last["EMA50"]) or (prev["EMA20"] >= prev["EMA50"] and last["EMA20"] < last["EMA50"])

    cond_rsi_buy  = last["RSI14"] < 38 and last["RSI14"] > prev["RSI14"]  # slightly looser to catch reversals
    cond_rsi_sell = last["RSI14"] > 62 and last["RSI14"] < prev["RSI14"]

    cond_macd_buy  = last["MACD"] > last["MACD_SIGNAL"] and last["MACD_HIST"] > prev["MACD_HIST"]
    cond_macd_sell = last["MACD"] < last["MACD_SIGNAL"] and last["MACD_HIST"] < prev["MACD_HIST"]

    # squeeze + breakout
    recent_bw = df["BB_WIDTH"].tail(40)
    squeeze = recent_bw.mean() < df["BB_WIDTH"].quantile(0.4)
    breakout_up = last["close"] > last["BB_MID"]
    breakout_down = last["close"] < last["BB_MID"]

    # Volume booster
    vol_boost = float(last.get("VOL_BOOST", 1.0) or 1.0)
    vol_factor = 1.0
    if vol_boost > 1.25:
        vol_factor = min(1.15, 1.0 + (vol_boost - 1.0) * 0.1)  # mild boost

    buy_score = 0.0
    sell_score = 0.0
    if cond_trend_buy:  buy_score  += 0.30
    if cond_rsi_buy:    buy_score  += 0.25
    if cond_macd_buy:   buy_score  += 0.25
    if squeeze and breakout_up:  buy_score += 0.20
    buy_score *= vol_factor

    if cond_trend_sell: sell_score += 0.30
    if cond_rsi_sell:   sell_score += 0.25
    if cond_macd_sell:  sell_score += 0.25
    if squeeze and breakout_down: sell_score += 0.20
    sell_score *= vol_factor

    verdict = "WATCH"
    if buy_score >= 0.60 and buy_score > sell_score:
        verdict = "BUY"
    elif sell_score >= 0.60 and sell_score > buy_score:
        verdict = "SELL"

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
    if cond_trend_buy:  reasons.append("EMA trend bullish")
    if cond_rsi_buy:    reasons.append("RSI recovering")
    if cond_macd_buy:   reasons.append("MACD bullish")
    if squeeze and breakout_up: reasons.append("Squeeze → breakout up")
    if cond_trend_sell: reasons.append("EMA trend bearish")
    if cond_rsi_sell:   reasons.append("RSI cooling")
    if cond_macd_sell:  reasons.append("MACD bearish")
    if squeeze and breakout_down: reasons.append("Squeeze → breakout down")
    if vol_boost > 1.25: reasons.append("Volume > median")

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

def analyze_symbol_multi(symbol: str, tfs=("1h","4h","1d")):
    """Multi-timeframe confirmation: require alignment in majority of TFs."""
    per_tf = {}
    buy_votes = sell_votes = 0
    avg_buy = avg_sell = 0.0
    price = None
    sl = tp = None

    for tf in tfs:
        df = fetch_ohlcv_df(symbol, tf, limit=400 if tf!="1d" else 250)
        df = compute_indicators(df)
        res = score_single_timeframe(df)
        per_tf[tf] = res
        if price is None: price = res["price"]
        avg_buy += res["buy_score"]
        avg_sell += res["sell_score"]
        if res["verdict"] == "BUY":  buy_votes += 1
        if res["verdict"] == "SELL": sell_votes += 1

    avg_buy = round(avg_buy/len(tfs), 2)
    avg_sell = round(avg_sell/len(tfs), 2)

    # confirmation rule: majority vote AND higher avg score
    final = "WATCH"
    if buy_votes >= 2 and avg_buy >= 0.6 and avg_buy > avg_sell:
        final = "BUY"
        # take SL/TP from shortest tf (1h) if present
        sl = per_tf[tfs[0]]["stop_loss"]
        tp = per_tf[tfs[0]]["target"]
    elif sell_votes >= 2 and avg_sell >= 0.6 and avg_sell > avg_buy:
        final = "SELL"
        sl = per_tf[tfs[0]]["stop_loss"]
        tp = per_tf[tfs[0]]["target"]

    return {
        "symbol": normalize_symbol(symbol),
        "timeframes": tfs,
        "price": price,
        "avg_buy": avg_buy,
        "avg_sell": avg_sell,
        "verdict": final,
        "sl": sl, "tp": tp,
        "per_tf": per_tf,
    }

# ----------- Telegram Handlers -----------
def pair_keyboard(pairs):
    buttons = []
    row = []
    for i, p in enumerate(pairs, 1):
        row.append(InlineKeyboardButton(p, callback_data=f"sig:{p}:1h"))
        if i % 3 == 0:
            buttons.append(row); row = []
    if row: buttons.append(row)
    return InlineKeyboardMarkup(buttons)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "🤖 Bitget Multi-TF Signal Bot\n\n"
        "Commands:\n"
        "/signal SYMBOL [tf] — analyze one pair (e.g. /signal BTC/USDT 1h)\n"
        "/signal — show quick pair buttons\n"
        "/scan [tf] — rank your watchlist by multi-TF strength\n"
        "/watchlist — show pairs I’m watching\n"
        "/config — settings\n\n"
        "Multi-timeframe confirmation (1h+4h+1d), volume filter, ATR SL/TP.\n"
        "Signals are informational only — not financial advice."
    )
    await update.message.reply_text(txt)

async def config_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (f"Default TF: {DEFAULT_INTERVAL}\n"
           f"Watchlist: {len(WATCHLIST)} pairs\n"
           f"Auto alerts: every {ALERT_FREQUENCY_MIN} min to your DM" + (f" (chat {USER_CHAT_ID})" if USER_CHAT_ID else ""))
    await update.message.reply_text(msg)

async def watchlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Watchlist:\n" + ", ".join(WATCHLIST))

async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # If no args → show inline suggestions
    if not context.args:
        await update.message.reply_text(
            "Pick a pair:", reply_markup=pair_keyboard(WATCHLIST[:12])
        )
        return
    symbol = context.args[0]
    tf = context.args[1] if len(context.args) > 1 else DEFAULT_INTERVAL
    await do_signal_reply(update, context, symbol, tf)

async def signal_button_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    try:
        _, symbol, tf = q.data.split(":")
    except Exception:
        await q.edit_message_text("Bad selection.")
        return
    # Replace message with analysis
    text = await do_signal_text(symbol, tf)
    await q.edit_message_text(text)

async def do_signal_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str, tf: str):
    try:
        text = await do_signal_text(symbol, tf)
        await update.message.reply_text(text)
    except Exception as e:
        logger.error("signal error: %s", traceback.format_exc())
        await update.message.reply_text(f"Error analyzing {symbol}: {e}")

async def do_signal_text(symbol: str, tf: str) -> str:
    # Multi-TF confirmation always uses ('1h','4h','1d'); display requested tf in header
    res = analyze_symbol_multi(symbol, tfs=("1h","4h","1d"))
    lines = []
    lines.append(f"📊 {normalize_symbol(symbol)} | req TF {tf} | MTF 1h/4h/1d")
    lines.append(f"Price: {res['price']:.8f}")
    lines.append(f"Avg Scores → Buy: {res['avg_buy']} | Sell: {res['avg_sell']}")
    lines.append(f"Verdict: {res['verdict']}")
    if res["sl"] and res["tp"]:
        lines.append(f"SL: {res['sl']}  TP: {res['tp']}")
    # concise per-TF reasons
    out = []
    for tf_i in res["timeframes"]:
        r = res["per_tf"][tf_i]
        out.append(f"{tf_i}: {r['verdict']} (B{r['buy_score']}/S{r['sell_score']}) | RSI {r['rsi']:.1f}")
    lines.append("TF detail → " + "  •  ".join(out))
    return "\n".join(lines)

async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tf = context.args[0] if context.args else DEFAULT_INTERVAL
    results = []
    for sym in WATCHLIST:
        try:
            r = analyze_symbol_multi(sym, tfs=("1h","4h","1d"))
            results.append(r)
            await asyncio.sleep(0.12)
        except Exception as e:
            logger.warning("scan fail %s: %s", sym, e)
    if not results:
        await update.message.reply_text("No data.")
        return
    # rank: prefer BUY with higher avg_buy, break ties by avg_sell
    ranked = sorted(results, key=lambda x: (x["verdict"]=="BUY", x["avg_buy"], -x["avg_sell"]), reverse=True)
    top = ranked[:10]
    lines = [f"🔎 Top setups (MTF 1h/4h/1d)"]
    for r in top:
        line = f"{r['symbol']} → {r['verdict']} | Avg B{r['avg_buy']} S{r['avg_sell']} | Price {r['price']:.6f}"
        if r["sl"] and r["tp"]:
            line += f" | SL {r['sl']} TP {r['tp']}"
        lines.append(line)
    await update.message.reply_text("\n".join(lines))

# ----------- Auto hourly alerts -----------
async def hourly_push(context: ContextTypes.DEFAULT_TYPE):
    if not USER_CHAT_ID:
        logger.info("USER_CHAT_ID not set; skipping auto alerts.")
        return
    try:
        results = []
        for sym in WATCHLIST:
            r = analyze_symbol_multi(sym, tfs=("1h","4h","1d"))
            results.append(r)
            await asyncio.sleep(0.12)
        # strong only: BUY with avg_buy >= 0.70
        strong = [r for r in results if r["verdict"]=="BUY" and r["avg_buy"]>=0.70]
        strong = sorted(strong, key=lambda x: (x["avg_buy"], -x["avg_sell"]), reverse=True)[:5]
        if not strong:
            msg = "⏰ Hourly scan: no high-conviction setups right now."
        else:
            lines = ["⏰ Hourly high-conviction setups:"]
            for r in strong:
                line = f"{r['symbol']} → BUY | Avg B{r['avg_buy']} S{r['avg_sell']} | {r['price']:.6f}"
                if r["sl"] and r["tp"]:
                    line += f" | SL {r['sl']} TP {r['tp']}"
                lines.append(line)
            msg = "\n".join(lines)
        await context.bot.send_message(chat_id=int(USER_CHAT_ID), text=msg)
    except Exception as e:
        logger.error("hourly_push error: %s", traceback.format_exc())

# ----------- Main -----------
def build_and_run():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", start_cmd))
    app.add_handler(CommandHandler("config", config_cmd))
    app.add_handler(CommandHandler("watchlist", watchlist_cmd))
    app.add_handler(CommandHandler("signal", signal_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.add_handler(CallbackQueryHandler(signal_button_cb, pattern=r"^sig:"))

    # hourly repeating job
    try:
        app.job_queue.run_repeating(hourly_push, interval=ALERT_FREQUENCY_MIN*60, first=60)
    except Exception as e:
        logger.warning("job_queue schedule failed: %s", e)

    logger.info("Starting Telegram bot (polling)...")
    app.run_polling()

if __name__ == "__main__":
    build_and_run()
