# Trading Bot (Signal Advisor)

This bot does **NOT place trades** – it only analyzes and tells you the best signals.

## Features
- RSI, EMA50/200, MACD, Bollinger Bands, ATR Stop Loss/Take Profit
- /signal SYMBOL [interval] → get signal for coin
- /scan [interval] → scan WATCHLIST coins

## Setup
1. Fork repo & deploy to Railway
2. Add Environment Variables:
   - TELEGRAM_BOT_TOKEN = your bot token
   - DEFAULT_INTERVAL = 15m
   - WATCHLIST = BTCUSDT,ETHUSDT,SOLUSDT
3. Deploy and use bot on Telegram.
