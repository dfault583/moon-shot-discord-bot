# moon-shot-discord-bot

Discord bot that posts stock/crypto prices and charts (candlesticks, volume,
moving averages). Single-file app in `bot.py`.

## Deployment

- **Hosted on Render**, with **Auto-Deploy: On Commit**.
- Render builds from the **`main`** branch. Any push to `main` triggers an
  automatic build + redeploy (runs `python bot.py` via the `Dockerfile`).

## Working convention

- **Commit and push fixes directly to `main`.** This is the deploy branch, so
  merging there is what ships the change. No separate feature branch or PR is
  required unless explicitly requested.
- After pushing to `main`, Render auto-deploys — no manual deploy step needed.
- To verify a pricing change, run the price command in Discord and compare
  against a reference quote source.

## Notes on price logic (`bot.py`)

- Data comes from yfinance `fast_info` / `ticker.info`.
- During pre-market, `last_price` is the most recent regular-session close and
  `previous_close` is the session before that. Pre-market change must be
  measured against `last_price`, not `previous_close`. After-hours change is
  measured against `last_price`.
