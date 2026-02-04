import discord
from discord.ext import commands
import yfinance as yf
import mplfinance as mpf
import io
import os
import numpy as np

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

def calculate_vwap(df):
          tp = (df['High'] + df['Low'] + df['Close']) / 3
          return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

def make_chart(df, symbol, timeframe):
          df['SMA20'] = df['Close'].rolling(window=20).mean()
          df['SMA50'] = df['Close'].rolling(window=50).mean()
          df['SMA200'] = df['Close'].rolling(window=200).mean()
          df['VWAP'] = calculate_vwap(df)
          mc = mpf.make_marketcolors(up='#00ff00', down='#ff00ff', edge='inherit', wick='inherit', volume={'up': '#00ff00', 'down': '#ff00ff'})
          s = mpf.make_mpf_style(marketcolors=mc, base_mpl_style='dark_background', facecolor='#0d1117', figcolor='#0d1117', gridstyle='-', gridcolor='#30363d', y_on_right=True)
          plots = [mpf.make_addplot(df['SMA20'], color='#00bfff', width=1), mpf.make_addplot(df['SMA50'], color='#ff1493', width=1), mpf.make_addplot(df['SMA200'], color='#ffa500', width=1.5), mpf.make_addplot(df['VWAP'], color='#ffff00', width=1, linestyle='--')]
          buf = io.BytesIO()
          mpf.plot(df, type='candle', style=s, volume=True, addplot=plots, savefig=dict(fname=buf, dpi=150, bbox_inches='tight'), title=f'{symbol} - {timeframe}')
          buf.seek(0)
          return buf

@bot.command(name='c')
async def chart_default(ctx, symbol: str = 'AAPL'):
          await ctx.send(f"Generating weekly chart for {symbol}...")
          ticker = yf.Ticker(symbol)
          df = ticker.history(period='1y', interval='1wk')
          buf = make_chart(df, symbol, 'Weekly')
          await ctx.send(file=discord.File(buf, filename=f"{symbol}_chart.png"))

@bot.command(name='cm')
async def chart_minute(ctx, minutes: int = 5, symbol: str = 'AAPL'):
          await ctx.send(f"Generating {minutes}min chart for {symbol}...")
          ticker = yf.Ticker(symbol)
          df = ticker.history(period='1d', interval=f'{minutes}m')
          buf = make_chart(df, symbol, f'{minutes}min')
          await ctx.send(file=discord.File(buf, filename=f"{symbol}_chart.png"))

@bot.command(name='cd')
async def chart_daily(ctx, symbol: str = 'AAPL'):
          await ctx.send(f"Generating daily chart for {symbol}...")
          ticker = yf.Ticker(symbol)
          df = ticker.history(period='6mo', interval='1d')
          buf = make_chart(df, symbol, 'Daily')
          await ctx.send(file=discord.File(buf, filename=f"{symbol}_chart.png"))

@bot.command(name='price')
async def price(ctx, symbol: str = 'AAPL'):
          ticker = yf.Ticker(symbol)
          info = ticker.info
          current = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
          prev_close = info.get('previousClose', 'N/A')
          high = info.get('dayHigh', 'N/A')
          low = info.get('dayLow', 'N/A')
          change = round(current - prev_close, 2) if current != 'N/A' and prev_close != 'N/A' else 'N/A'
          pct = round((change / prev_close) * 100, 2) if change != 'N/A' else 'N/A'
          emoji = "🟢" if change != 'N/A' and change >= 0 else "🔴"
          await ctx.send(f"**{symbol.upper()}** {emoji}\nPrice: ${current}\nChange: ${change} ({pct}%)\nHigh: ${high} | Low: ${low}")

@bot.command(name='help_bot')
async def help_bot(ctx):
          await ctx.send("**Clawdussy Commands:**\n`!c [symbol]` - Weekly chart\n`!cm [mins] [symbol]` - Minute chart (ex: !cm 5 TSLA)\n`!cd [symbol]` - Daily chart\n`!price [symbol]` - Current price")

@bot.event
async def on_ready():
          print(f'{bot.user} is online!')

bot.run(os.getenv('DISCORD_TOKEN'))
