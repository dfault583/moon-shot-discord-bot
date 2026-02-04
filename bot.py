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
      try:
                if df is None or df.empty or len(df) < 2:
                              return None
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
except Exception:
        return None

@bot.command(name='c')
async def chart_default(ctx, symbol: str = 'AAPL'):
      await ctx.send(f"Generating weekly chart for {symbol}...")
      ticker = yf.Ticker(symbol)
      df = ticker.history(period='1y', interval='1wk')
      buf = make_chart(df, symbol, '1wk')
      if buf is None:
                await ctx.send(f"No data for {symbol}. Check symbol or try later.")
                return
            await ctx.send(file=discord.File(buf, f'{symbol}_chart.png'))

@bot.command(name='cm')
async def chart_minute(ctx, minutes: int = 5, symbol: str = 'AAPL'):
      await ctx.send(f"Generating {minutes}min chart for {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='5d', interval=f'{minutes}m')
    buf = make_chart(df, symbol, f'{minutes}min')
    if buf is None:
              await ctx.send(f"No intraday data for {symbol}. Market closed or symbol invalid.")
              return
          await ctx.send(file=discord.File(buf, f'{symbol}_chart.png'))

@bot.command(name='cd')
async def chart_daily(ctx, symbol: str = 'AAPL'):
      await ctx.send(f"Generating daily chart for {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='6mo', interval='1d')
    buf = make_chart(df, symbol, '1d')
    if buf is None:
              await ctx.send(f"No data for {symbol}. Check symbol or try later.")
              return
          await ctx.send(file=discord.File(buf, f'{symbol}_chart.png'))

@bot.command(name='price')
async def price_info(ctx, symbol: str = 'AAPL'):
      ticker = yf.Ticker(symbol)
    info = ticker.info
    price = info.get('regularMarketPrice', 'N/A')
    change = info.get('regularMarketChangePercent', 0)
    await ctx.send(f"**{symbol}**: ${price} ({change:+.2f}%)")

@bot.command(name='help_bot')
async def help_bot(ctx):
      msg = "**Commands:**\n!c [symbol] - Weekly chart\n!cm [minutes] [symbol] - Minute chart\n!cd [symbol] - Daily chart\n!price [symbol] - Price info"
    await ctx.send(msg)

@bot.event
async def on_ready():
      print(f'{bot.user} is connected!')

bot.run(os.environ['DISCORD_TOKEN'])
