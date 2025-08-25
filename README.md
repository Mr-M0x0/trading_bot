# Bitget Signal Bot (Telegram)

Signals only. Uses Bitget public market data via ccxt.

## Files
- bot.py
- requirements.txt
- Procfile

## Deploy (Render)
1. Push repo to GitHub.
2. On Render: New → Web Service → Connect GitHub → choose this repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `python bot.py`
5. Environment variables (Render > Environment):
   - TELEGRAM_BOT_TOKEN = (from BotFather)
   - DEFAULT_INTERVAL = 15m
   - WATCHLIST = BTC/USDT,ETH/USDT,...
   - (optional) BITGET_API_KEY, BITGET_API_SECRET, BITGET_API_PASSPHRASE
6. Deploy and use Telegram commands:
   - `/start` `/signal SYMBOL [interval]` `/scan [interval]` `/watchlist` `/config`
