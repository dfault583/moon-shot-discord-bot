import discord
from discord.ext import commands
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import mplfinance as mpf
import io
import os
import numpy as np
from datetime import datetime, timedelta

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize Alpaca client
ALPACA_API_KEY = os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def calculate_vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()

def make_chart(df, symbol, timeframe):
    try:
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        df['SMA200'] = df['close'].rolling(window=200).mean()
        df['VWAP'] = calculate_vwap(df)
        mc = mpf.make_marketcolors(up='#00ff00', down='#ff00ff', edge='inherit', wick='inherit', volume={'up': '#00ff00', 'down': '#ff00ff'})
        s = mpf.make_mpf_style(marketcolors=mc, base_mpl_style='dark_background', facecolor='#0d1117', figcolor='#0d1117', gridstyle='-', gridcolor='#30363d', y_on_right=True)
        plots = [
            mpf.make_addplot(df['SMA20'], color='#00bfff', width=1),
            mpf.make_addplot(df['SMA50'], color='#ff1493', width=1),
            mpf.make_addplot(df['SMA200'], color='#ffa500', width=1.5),
            mpf.make_addplot(df['VWAP'], color='#ffff00', width=1, linestyle='--')
        ]
        buf = io.BytesIO()
        mpf.plot(df, type='candle', style=s, volume=True, addplot=plots, savefig=dict(fname=buf, dpi=150, bbox_inches='tight'), title=f'{symbol} - {timeframe}')
        buf.seek(0)
        return buf
    except:
        return None

@bot.command(name='c')
async def chart_default(ctx, symbol: str = 'AAPL'):
    await ctx.send(f"Generating weekly chart for {symbol}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    request = StockBarsRequest(symbol_or_symbols=symbol.upper(), timeframe=TimeFrame.Week, start=start_date, end=end_date)
    bars = stock_client.get_stock_bars(request)
    df = bars.df
    if symbol.upper() in df.index.get_level_values('symbol'):
        df = df.xs(symbol.upper(), level='symbol')
    df.index = df.index.tz_localize(None)
    df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
    buf = make_chart(df, symbol.upper(), '1wk')
    if buf is None:
        await ctx.send(f"No data for {symbol}.")
        return
    await ctx.send(file=discord.File(buf, filename=f'{symbol}_weekly.png'))

@bot.command(name='cd')
async def chart_daily(ctx, symbol: str = 'AAPL'):
    await ctx.send(f"Generating daily chart for {symbol}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    request = StockBarsRequest(symbol_or_symbols=symbol.upper(), timeframe=TimeFrame.Day, start=start_date, end=end_date)
    bars = stock_client.get_stock_bars(request)
    df = bars.df
    if symbol.upper() in df.index.get_level_values('symbol'):
        df = df.xs(symbol.upper(), level='symbol')
    df.index = df.index.tz_localize(None)
    df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
    buf = make_chart(df, symbol.upper(), '1d')
    if buf is None:
        await ctx.send(f"No data for {symbol}.")
        return
    await ctx.send(file=discord.File(buf, filename=f'{symbol}_daily.png'))

@bot.command(name='ch')
async def chart_hourly(ctx, symbol: str = 'AAPL'):
    await ctx.send(f"Generating hourly chart for {symbol}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    request = StockBarsRequest(symbol_or_symbols=symbol.upper(), timeframe=TimeFrame.Hour, start=start_date, end=end_date)
    bars = stock_client.get_stock_bars(request)
    df = bars.df
    if symbol.upper() in df.index.get_level_values('symbol'):
        df = df.xs(symbol.upper(), level='symbol')
    df.index = df.index.tz_localize(None)
    df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
    buf = make_chart(df, symbol.upper(), '1h')
    if buf is None:
        await ctx.send(f"No data for {symbol}.")
        return
    await ctx.send(file=discord.File(buf, filename=f'{symbol}_hourly.png'))

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

bot.run(os.getenv('DISCORD_TOKEN'))
