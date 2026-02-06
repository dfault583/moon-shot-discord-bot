import discord
from discord.ext import commands
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import os
import numpy as np
from datetime import datetime, timedelta
import pytz

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)
bot.remove_command('help')

# Initialize Alpaca client
ALPACA_API_KEY = os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# TradingView color scheme
TV_BG = '#131722'
TV_CANDLE_UP = '#26a69a'
TV_CANDLE_DOWN = '#ef5350'
TV_GRID = '#1e222d'
TV_TEXT = '#d1d4dc'
TV_BORDER = '#2a2e39'
TV_VOL_UP = '#26a69a'
TV_VOL_DOWN = '#ef5350'

def calculate_vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()

def make_chart(df, symbol, timeframe):
    try:
        if len(df) < 2:
            print(f"Not enough data for {symbol} {timeframe}: {len(df)} rows")
            return None

        plots = []

        if len(df) >= 20:
            df['SMA20'] = df['close'].rolling(window=20).mean()
            plots.append(mpf.make_addplot(df['SMA20'], color='#2962ff', width=1.2))
        if len(df) >= 50:
            df['SMA50'] = df['close'].rolling(window=50).mean()
            plots.append(mpf.make_addplot(df['SMA50'], color='#ff6d00', width=1.2))
        if len(df) >= 200:
            df['SMA200'] = df['close'].rolling(window=200).mean()
            plots.append(mpf.make_addplot(df['SMA200'], color='#ab47bc', width=1.5))

        df['VWAP'] = calculate_vwap(df)
        plots.append(mpf.make_addplot(df['VWAP'], color='#ffeb3b', width=1, linestyle='--'))

        mc = mpf.make_marketcolors(
            up=TV_CANDLE_UP, down=TV_CANDLE_DOWN,
            edge={'up': TV_CANDLE_UP, 'down': TV_CANDLE_DOWN},
            wick={'up': TV_CANDLE_UP, 'down': TV_CANDLE_DOWN},
            volume={'up': TV_VOL_UP, 'down': TV_VOL_DOWN},
            ohlc={'up': TV_CANDLE_UP, 'down': TV_CANDLE_DOWN}
        )
        s = mpf.make_mpf_style(
            marketcolors=mc,
            facecolor=TV_BG,
            figcolor=TV_BG,
            gridcolor=TV_GRID,
            gridstyle='-',
            gridaxis='both',
            y_on_right=True,
            rc={
                'font.size': 9,
                'axes.labelcolor': TV_TEXT,
                'axes.edgecolor': TV_BORDER,
                'xtick.color': TV_TEXT,
                'ytick.color': TV_TEXT,
                'text.color': TV_TEXT,
                'figure.titlesize': 12,
                'axes.titlesize': 12,
            }
        )

        last_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close
        change = last_close - prev_close
        pct_change = (change / prev_close) * 100
        sign = '+' if change >= 0 else ''
        title = f'{symbol} {timeframe}  {last_close:.2f}  {sign}{change:.2f} ({sign}{pct_change:.2f}%)'

        buf = io.BytesIO()
        fig, axes = mpf.plot(
            df, type='candle', style=s, volume=True,
            addplot=plots,
            figsize=(12, 7),
            tight_layout=True,
            scale_padding={'left': 0.05, 'top': 0.6, 'right': 1.0, 'bottom': 0.5},
            returnfig=True,
            volume_panel=1,
            panel_ratios=(3, 1)
        )
        fig.suptitle(title, color=TV_TEXT, fontsize=13, fontweight='bold', x=0.08, ha='left')
        for ax in axes:
            ax.set_facecolor(TV_BG)
            ax.tick_params(colors=TV_TEXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(TV_BORDER)
        fig.savefig(buf, dpi=150, bbox_inches='tight', facecolor=TV_BG, edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Chart error: {e}")
        import traceback
        traceback.print_exc()
        return None

@bot.command(name='help')
async def help_command(ctx):
    embed = discord.Embed(
        title='Moon Shot Commands',
        description='Stock charting bot powered by Alpaca',
        color=0x26a69a
    )
    embed.add_field(
        name='Charts',
        value=(
            '**!cw SYMBOL** — Weekly chart (1 year)\n'
            '**!cd SYMBOL** — Daily chart (3 months)\n'
            '**!ch SYMBOL** — Hourly chart (5 days)\n'
            '**!cm1 SYMBOL** — 1 min chart (today)\n'
            '**!cm5 SYMBOL** — 5 min chart (today)\n'
            '**!cm15 SYMBOL** — 15 min chart (today)\n'
            '**!cm30 SYMBOL** — 30 min chart (today)'
        ),
        inline=False
    )
    embed.add_field(
        name='Overlays',
        value='SMA 20 / 50 / 200 + VWAP',
        inline=False
    )
    embed.set_footer(text='Default symbol: AAPL')
    await ctx.send(embed=embed)

@bot.command(name='cw')
async def chart_weekly(ctx, symbol: str = 'AAPL'):
    try:
        await ctx.send(f"Generating weekly chart for {symbol.upper()}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        request = StockBarsRequest(symbol_or_symbols=symbol.upper(), timeframe=TimeFrame.Week, start=start_date, end=end_date, feed='iex')
        bars = stock_client.get_stock_bars(request)
        df = bars.df
        if symbol.upper() in df.index.get_level_values('symbol'):
            df = df.xs(symbol.upper(), level='symbol')
        df.index = df.index.tz_localize(None)
        buf = make_chart(df, symbol.upper(), '1W')
        if buf is None:
            await ctx.send(f"No data for {symbol.upper()}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_weekly.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='cd')
async def chart_daily(ctx, symbol: str = 'AAPL'):
    try:
        await ctx.send(f"Generating daily chart for {symbol.upper()}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        request = StockBarsRequest(symbol_or_symbols=symbol.upper(), timeframe=TimeFrame.Day, start=start_date, end=end_date, feed='iex')
        bars = stock_client.get_stock_bars(request)
        df = bars.df
        if symbol.upper() in df.index.get_level_values('symbol'):
            df = df.xs(symbol.upper(), level='symbol')
        df.index = df.index.tz_localize(None)
        buf = make_chart(df, symbol.upper(), '1D')
        if buf is None:
            await ctx.send(f"No data for {symbol.upper()}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_daily.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='ch')
async def chart_hourly(ctx, symbol: str = 'AAPL'):
    try:
        await ctx.send(f"Generating hourly chart for {symbol.upper()}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        request = StockBarsRequest(symbol_or_symbols=symbol.upper(), timeframe=TimeFrame.Hour, start=start_date, end=end_date, feed='iex')
        bars = stock_client.get_stock_bars(request)
        df = bars.df
        if symbol.upper() in df.index.get_level_values('symbol'):
            df = df.xs(symbol.upper(), level='symbol')
        df.index = df.index.tz_localize(None)
        buf = make_chart(df, symbol.upper(), '1H')
        if buf is None:
            await ctx.send(f"No data for {symbol.upper()}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_hourly.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

async def _chart_minute(ctx, symbol, minutes):
    try:
        await ctx.send(f"Generating {minutes}min chart for {symbol.upper()}...")
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        today_start = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
        if now_et.hour < 4:
            today_start = today_start - timedelta(days=1)

        request = StockBarsRequest(
            symbol_or_symbols=symbol.upper(),
            timeframe=TimeFrame.Minute,
            start=today_start,
            end=now_et,
            feed='iex'
        )
        bars = stock_client.get_stock_bars(request)
        df = bars.df
        if len(df) == 0:
            await ctx.send(f"No data for {symbol.upper()} today. Market may be closed.")
            return
        if symbol.upper() in df.index.get_level_values('symbol'):
            df = df.xs(symbol.upper(), level='symbol')
        if minutes > 1:
            df = df.resample(f'{minutes}min').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()
        df.index = df.index.tz_localize(None)
        if len(df) < 2:
            await ctx.send(f"Not enough data for {symbol.upper()} today yet.")
            return
        buf = make_chart(df, symbol.upper(), f'{minutes}m')
        if buf is None:
            await ctx.send(f"No data for {symbol.upper()}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_{minutes}min.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='cm1')
async def chart_1min(ctx, symbol: str = 'AAPL'):
    await _chart_minute(ctx, symbol, 1)

@bot.command(name='cm5')
async def chart_5min(ctx, symbol: str = 'AAPL'):
    await _chart_minute(ctx, symbol, 5)

@bot.command(name='cm15')
async def chart_15min(ctx, symbol: str = 'AAPL'):
    await _chart_minute(ctx, symbol, 15)

@bot.command(name='cm30')
async def chart_30min(ctx, symbol: str = 'AAPL'):
    await _chart_minute(ctx, symbol, 30)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

bot.run(os.getenv('DISCORD_TOKEN'))
