import discord
from discord.ext import commands
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D
import yfinance as yf
import io
import os
import numpy as np
import pandas as pd
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

# Common crypto aliases
CRYPTO_ALIASES = {
    'BTC': 'BTC-USD', 'BITCOIN': 'BTC-USD',
    'ETH': 'ETH-USD', 'ETHEREUM': 'ETH-USD',
    'SOL': 'SOL-USD', 'SOLANA': 'SOL-USD',
    'XRP': 'XRP-USD', 'RIPPLE': 'XRP-USD',
    'DOGE': 'DOGE-USD', 'DOGECOIN': 'DOGE-USD',
    'ADA': 'ADA-USD', 'CARDANO': 'ADA-USD',
    'AVAX': 'AVAX-USD', 'AVALANCHE': 'AVAX-USD',
    'DOT': 'DOT-USD', 'POLKADOT': 'DOT-USD',
    'MATIC': 'MATIC-USD', 'POLYGON': 'MATIC-USD',
    'LINK': 'LINK-USD', 'CHAINLINK': 'LINK-USD',
    'SHIB': 'SHIB-USD',
    'LTC': 'LTC-USD', 'LITECOIN': 'LTC-USD',
    'UNI': 'UNI-USD', 'UNISWAP': 'UNI-USD',
    'ATOM': 'ATOM-USD', 'COSMOS': 'ATOM-USD',
    'XLM': 'XLM-USD', 'STELLAR': 'XLM-USD',
    'ALGO': 'ALGO-USD', 'ALGORAND': 'ALGO-USD',
    'NEAR': 'NEAR-USD',
    'APT': 'APT-USD', 'APTOS': 'APT-USD',
    'ARB': 'ARB-USD', 'ARBITRUM': 'ARB-USD',
    'OP': 'OP-USD', 'OPTIMISM': 'OP-USD',
    'SUI': 'SUI-USD',
    'PEPE': 'PEPE-USD',
    'WIF': 'WIF-USD',
    'BONK': 'BONK-USD',
    'RENDER': 'RENDER-USD',
    'FET': 'FET-USD',
    'INJ': 'INJ-USD', 'INJECTIVE': 'INJ-USD',
    'TIA': 'TIA-USD', 'CELESTIA': 'TIA-USD',
    'SEI': 'SEI-USD',
    'JUP': 'JUP-USD', 'JUPITER': 'JUP-USD',
}

def resolve_crypto_symbol(symbol):
    """Convert user input to yfinance crypto ticker format."""
    symbol = symbol.upper().strip()
    if symbol in CRYPTO_ALIASES:
        return CRYPTO_ALIASES[symbol]
    if symbol.endswith('-USD'):
        return symbol
    return f'{symbol}-USD'

# Timeframe mappings for yfinance
YF_INTERVALS = {
    'month': '1mo',
    'week': '1wk',
    'day': '1d',
    'hour': '1h',
    '1min': '1m',
    '5min': '5m',
    '15min': '15m',
    '30min': '30m',
}

def get_bars_alpaca(symbol, timeframe, start_date, end_date):
    """Try to get data from Alpaca first."""
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            feed='iex'
        )
        bars = stock_client.get_stock_bars(request)
        df = bars.df
        if len(df) == 0:
            return None
        if symbol in df.index.get_level_values('symbol'):
            df = df.xs(symbol, level='symbol')
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"Alpaca error for {symbol}: {e}")
        return None

def get_bars_yfinance(symbol, interval, period=None, start_date=None, end_date=None):
    """Fallback to yfinance for OTC and unsupported tickers."""
    try:
        ticker = yf.Ticker(symbol)
        if period:
            df = ticker.history(period=period, interval=interval)
        else:
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        if df is None or len(df) == 0:
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={'stock splits': 'stock_splits'})
        # Keep only OHLCV columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]
        df.index.name = 'timestamp'
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"yfinance error for {symbol}: {e}")
        return None

def get_bars(symbol, alpaca_tf, yf_interval, start_date, end_date, yf_period=None):
    """Try Alpaca first, fall back to yfinance."""
    df = get_bars_alpaca(symbol, alpaca_tf, start_date, end_date)
    if df is not None and len(df) > 0:
        return df, 'alpaca'
    print(f"Alpaca returned no data for {symbol}, trying yfinance...")
    df = get_bars_yfinance(symbol, yf_interval, period=yf_period, start_date=start_date, end_date=end_date)
    if df is not None and len(df) > 0:
        return df, 'yfinance'
    return None, None

def get_crypto_bars(symbol, yf_interval, period=None, start_date=None, end_date=None):
    """Get crypto data from yfinance."""
    ticker_symbol = resolve_crypto_symbol(symbol)
    df = get_bars_yfinance(ticker_symbol, yf_interval, period=period, start_date=start_date, end_date=end_date)
    if df is not None and len(df) > 0:
        return df
    return None


def get_price_info(symbol):
    """Get current price info for a stock or crypto, including pre/post market."""
    symbol = symbol.upper().strip()

    # Try as stock first
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        if info and hasattr(info, 'last_price') and info.last_price is not None:
            result = _build_price_embed(ticker, symbol, is_crypto=False)
            if result:
                return result
    except Exception as e:
        print(f"Stock lookup failed for {symbol}: {e}")

    # Try as crypto
    crypto_ticker_str = resolve_crypto_symbol(symbol)
    try:
        ticker = yf.Ticker(crypto_ticker_str)
        info = ticker.fast_info
        if info and hasattr(info, 'last_price') and info.last_price is not None:
            result = _build_price_embed(ticker, crypto_ticker_str, is_crypto=True)
            if result:
                return result
    except Exception as e:
        print(f"Crypto lookup failed for {crypto_ticker_str}: {e}")

    return None

def _build_price_embed(ticker, symbol, is_crypto=False):
    """Build a Discord embed with price info."""
    try:
        info = ticker.fast_info
        display_name = symbol.replace('-USD', '') if is_crypto else symbol

        last_price = info.last_price
        prev_close = info.previous_close if hasattr(info, 'previous_close') and info.previous_close else None
        open_price = info.open if hasattr(info, 'open') and info.open else None
        day_high = info.day_high if hasattr(info, 'day_high') and info.day_high else None
        day_low = info.day_low if hasattr(info, 'day_low') and info.day_low else None
        volume = info.last_volume if hasattr(info, 'last_volume') and info.last_volume else None

        # Calculate change from previous close
        change = 0
        pct_change = 0
        if prev_close and prev_close > 0:
            change = last_price - prev_close
            pct_change = (change / prev_close) * 100

        sign = '+' if change >= 0 else ''
        color = 0x26a69a if change >= 0 else 0xef5350

        # Determine market status and extended hours price
        market_status = ''
        extended_price = None
        extended_change = None
        extended_pct = None

        if not is_crypto:
            et = pytz.timezone('US/Eastern')
            now_et = datetime.now(et)
            hour = now_et.hour
            minute = now_et.minute
            weekday = now_et.weekday()  # 0=Mon, 6=Sun

            if weekday >= 5:
                market_status = 'Market Closed (Weekend)'
            elif hour < 4:
                market_status = 'Market Closed'
            elif hour < 9 or (hour == 9 and minute < 30):
                market_status = 'Pre-Market'
            elif hour < 16:
                market_status = 'Market Open'
            elif hour < 20:
                market_status = 'After Hours'
            else:
                market_status = 'Market Closed'

            # Get extended hours price data
            try:
                ext_info = ticker.info
                if market_status == 'Pre-Market':
                    pre_price = ext_info.get('preMarketPrice')
                    if pre_price and pre_price > 0 and prev_close and prev_close > 0:
                        extended_price = pre_price
                        extended_change = pre_price - prev_close
                        extended_pct = (extended_change / prev_close) * 100
                elif market_status == 'After Hours':
                    post_price = ext_info.get('postMarketPrice')
                    if post_price and post_price > 0:
                        extended_price = post_price
                        extended_change = post_price - last_price
                        extended_pct = (extended_change / last_price) * 100
            except Exception as e:
                print(f"Extended hours data error: {e}")
        else:
            market_status = '24/7 Market'
        # Build embed
        asset_type = 'Crypto' if is_crypto else 'Stock'
        embed = discord.Embed(
            title=f'{display_name}',
            color=color
        )

        # Main price line
        embed.add_field(
            name='Price',
            value=f'**${last_price:,.2f}** {sign}{change:,.2f} ({sign}{pct_change:.2f}%)',
            inline=False
        )

        # Extended hours price if available
        if extended_price is not None:
            ext_sign = '+' if extended_change >= 0 else ''
            ext_label = 'Pre-Market' if market_status == 'Pre-Market' else 'After Hours'
            embed.add_field(
                name=ext_label,
                value=f'**${extended_price:,.2f}** {ext_sign}{extended_change:,.2f} ({ext_sign}{extended_pct:.2f}%)',
                inline=False
            )

        # Details
        details = []
        if prev_close:
            details.append(f'Prev Close: ${prev_close:,.2f}')
        if open_price:
            details.append(f'Open: ${open_price:,.2f}')
        if day_high and day_low:
            details.append(f'Range: ${day_low:,.2f} - ${day_high:,.2f}')
        if volume:
            if volume >= 1_000_000:
                vol_str = f'{volume / 1_000_000:.2f}M'
            elif volume >= 1_000:
                vol_str = f'{volume / 1_000:.1f}K'
            else:
                vol_str = f'{volume:,}'
            details.append(f'Volume: {vol_str}')

        if details:
            embed.add_field(
                name='Details',
                value='\n'.join(details),
                inline=False
            )

        embed.set_footer(text=f'{asset_type} \u2022 {market_status}')
        return embed
    except Exception as e:
        print(f"Build price embed error: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()

def make_chart(df, symbol, timeframe, display_count=None, source=None):
    try:
        if len(df) < 2:
            print(f"Not enough data for {symbol} {timeframe}: {len(df)} rows")
            return None

        # Calculate indicators on FULL data first
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        df['SMA200'] = df['close'].rolling(window=200).mean()
        df['VWAP'] = calculate_vwap(df)

        # Trim to display window AFTER calculating indicators
        if display_count and len(df) > display_count:
            df = df.iloc[-display_count:]

        plots = []
        legend_items = []

        if df['SMA20'].notna().any():
            plots.append(mpf.make_addplot(df['SMA20'], color='#2962ff', width=1.2))
            legend_items.append(('SMA 20', '#2962ff', '-'))
        if df['SMA50'].notna().any():
            plots.append(mpf.make_addplot(df['SMA50'], color='#ff6d00', width=1.2))
            legend_items.append(('SMA 50', '#ff6d00', '-'))
        if df['SMA200'].notna().any():
            plots.append(mpf.make_addplot(df['SMA200'], color='#ab47bc', width=1.5))
            legend_items.append(('SMA 200', '#ab47bc', '-'))
        if df['VWAP'].notna().any():
            plots.append(mpf.make_addplot(df['VWAP'], color='#ffeb3b', width=1, linestyle='--'))
            legend_items.append(('VWAP', '#ffeb3b', '--'))

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
        src_tag = ' [YF]' if source == 'yfinance' else ''
        title = f'{symbol} {timeframe} {last_close:.2f} {sign}{change:.2f} ({sign}{pct_change:.2f}%){src_tag}'

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

        # Add indicator legend
        if legend_items:
            handles = []
            for name, color, ls in legend_items:
                handles.append(Line2D([0], [0], color=color, linewidth=1.5, linestyle=ls, label=name))
            axes[0].legend(
                handles=handles,
                loc='upper left',
                fontsize=8,
                facecolor=TV_BG,
                edgecolor=TV_BORDER,
                labelcolor=TV_TEXT,
                framealpha=0.8,
                borderpad=0.4,
                handlelength=1.5
            )

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
        description='Stock & Crypto charting bot powered by Alpaca + Yahoo Finance',
        color=0x26a69a
    )
    embed.add_field(
        name='Stock Charts',
        value=(
            '**!cm SYMBOL** \u2014 Monthly chart (2 years)\n'
            '**!cw SYMBOL** \u2014 Weekly chart (1 year)\n'
            '**!cd SYMBOL** \u2014 Daily chart (3 months)\n'
            '**!ch SYMBOL** \u2014 Hourly chart (5 days)\n'
            '**!cm1 SYMBOL** \u2014 1 min chart (today)\n'
            '**!cm5 SYMBOL** \u2014 5 min chart (today)\n'
            '**!cm15 SYMBOL** \u2014 15 min chart (today)\n'
            '**!cm30 SYMBOL** \u2014 30 min chart (today)'
        ),
        inline=False
    )
    embed.add_field(
        name='Crypto Charts',
        value=(
            '**!ccm SYMBOL** \u2014 Monthly chart (2 years)\n'
            '**!ccw SYMBOL** \u2014 Weekly chart (1 year)\n'
            '**!ccd SYMBOL** \u2014 Daily chart (3 months)\n'
            '**!cch SYMBOL** \u2014 Hourly chart (5 days)\n'
            '**!cc1 SYMBOL** \u2014 1 min chart (today)\n'
            '**!cc5 SYMBOL** \u2014 5 min chart (today)\n'
            '**!cc15 SYMBOL** \u2014 15 min chart (today)\n'
            '**!cc30 SYMBOL** \u2014 30 min chart (today)\n'
            '\nExamples: !cc BTC, !ccw ETH, !cch SOL'
        ),
        inline=False
    )
    embed.add_field(
        name='Price Check',
        value=(
            '**!p SYMBOL** \u2014 Quick price check (stocks & crypto)\n'
            'Shows current price, change, pre-market/after-hours data\n'
            'Example: !p AAPL, !p BTC, !p TSLA'
        ),
        inline=False
    )
    embed.add_field(
        name='Overlays',
        value='SMA 20 / 50 / 200 + VWAP',
        inline=False
    )
    embed.add_field(
        name='Data',
        value='Stocks: NYSE, NASDAQ & OTC (Alpaca + Yahoo Finance)\nCrypto: BTC, ETH, SOL, XRP, DOGE, ADA, and 30+ more via Yahoo Finance',
        inline=False
    )
    embed.set_footer(text='Default stock: AAPL | Default crypto: BTC')
    await ctx.send(embed=embed)

# ============================================================
# STOCK COMMANDS
# ============================================================

@bot.command(name='cm')
async def chart_monthly(ctx, symbol: str = 'AAPL'):
    try:
        symbol = symbol.upper()
        await ctx.send(f"Generating monthly chart for {symbol}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        df, source = get_bars(symbol, TimeFrame.Month, '1mo', start_date, end_date, yf_period='2y')
        if df is None:
            await ctx.send(f"No data found for {symbol}.")
            return
        buf = make_chart(df, symbol, '1M', source=source)
        if buf is None:
            await ctx.send(f"No data for {symbol}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_monthly.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='cw')
async def chart_weekly(ctx, symbol: str = 'AAPL'):
    try:
        symbol = symbol.upper()
        await ctx.send(f"Generating weekly chart for {symbol}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        df, source = get_bars(symbol, TimeFrame.Week, '1wk', start_date, end_date, yf_period='2y')
        if df is None:
            await ctx.send(f"No data found for {symbol}.")
            return
        buf = make_chart(df, symbol, '1W', display_count=52, source=source)
        if buf is None:
            await ctx.send(f"No data for {symbol}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_weekly.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='cd')
async def chart_daily(ctx, symbol: str = 'AAPL'):
    try:
        symbol = symbol.upper()
        await ctx.send(f"Generating daily chart for {symbol}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df, source = get_bars(symbol, TimeFrame.Day, '1d', start_date, end_date, yf_period='1y')
        if df is None:
            await ctx.send(f"No data found for {symbol}.")
            return
        buf = make_chart(df, symbol, '1D', display_count=65, source=source)
        if buf is None:
            await ctx.send(f"No data for {symbol}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_daily.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='ch')
async def chart_hourly(ctx, symbol: str = 'AAPL'):
    try:
        symbol = symbol.upper()
        await ctx.send(f"Generating hourly chart for {symbol}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        df, source = get_bars(symbol, TimeFrame.Hour, '1h', start_date, end_date, yf_period='5d')
        if df is None:
            await ctx.send(f"No data found for {symbol}.")
            return
        buf = make_chart(df, symbol, '1H', display_count=40, source=source)
        if buf is None:
            await ctx.send(f"No data for {symbol}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_hourly.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

async def _chart_minute(ctx, symbol, minutes):
    try:
        symbol = symbol.upper()
        await ctx.send(f"Generating {minutes}min chart for {symbol}...")
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        today_start = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
        if now_et.hour < 4:
            today_start = today_start - timedelta(days=1)

        # Try Alpaca first
        df = get_bars_alpaca(symbol, TimeFrame.Minute, today_start, now_et)
        source = 'alpaca'

        # Fallback to yfinance
        if df is None or len(df) == 0:
            print(f"Alpaca no minute data for {symbol}, trying yfinance...")
            yf_interval = f'{minutes}m' if minutes > 1 else '1m'
            df = get_bars_yfinance(symbol, yf_interval, period='1d')
            source = 'yfinance'

        if df is None or len(df) == 0:
            await ctx.send(f"No data for {symbol} today. Market may be closed.")
            return

        if minutes > 1 and source == 'alpaca':
            df = df.resample(f'{minutes}min').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if len(df) < 2:
            await ctx.send(f"Not enough data for {symbol} today yet.")
            return
        buf = make_chart(df, symbol, f'{minutes}m', source=source)
        if buf is None:
            await ctx.send(f"No data for {symbol}.")
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


# ============================================================
# PRICE COMMAND
# ============================================================

@bot.command(name='p')
async def price_check(ctx, symbol: str = 'AAPL'):
    """Quick price check for stocks and crypto."""
    try:
        symbol = symbol.upper().strip()
        embed = get_price_info(symbol)
        if embed is None:
            await ctx.send(f'No data found for {symbol}. Check the symbol and try again.')
            return
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f'Error: {e}')

# ============================================================
# CRYPTO COMMANDS
# ============================================================

@bot.command(name='ccd')
async def crypto_daily(ctx, symbol: str = 'BTC'):
    """Daily crypto chart (3 months)."""
    try:
        ticker = resolve_crypto_symbol(symbol)
        display_name = ticker.replace('-USD', '')
        await ctx.send(f"Generating daily chart for {display_name}...")
        df = get_crypto_bars(symbol, '1d', period='3mo')
        if df is None:
            await ctx.send(f"No data found for {display_name}. Check the symbol and try again.")
            return
        buf = make_chart(df, display_name, '1D', display_count=65, source='yfinance')
        if buf is None:
            await ctx.send(f"Could not generate chart for {display_name}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{display_name}_crypto_daily.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='ccm')
async def crypto_monthly(ctx, symbol: str = 'BTC'):
    """Monthly crypto chart (2 years)."""
    try:
        ticker = resolve_crypto_symbol(symbol)
        display_name = ticker.replace('-USD', '')
        await ctx.send(f"Generating monthly chart for {display_name}...")
        df = get_crypto_bars(symbol, '1mo', period='2y')
        if df is None:
            await ctx.send(f"No data found for {display_name}. Check the symbol and try again.")
            return
        buf = make_chart(df, display_name, '1M', source='yfinance')
        if buf is None:
            await ctx.send(f"Could not generate chart for {display_name}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{display_name}_crypto_monthly.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='ccw')
async def crypto_weekly(ctx, symbol: str = 'BTC'):
    """Weekly crypto chart (1 year)."""
    try:
        ticker = resolve_crypto_symbol(symbol)
        display_name = ticker.replace('-USD', '')
        await ctx.send(f"Generating weekly chart for {display_name}...")
        df = get_crypto_bars(symbol, '1wk', period='1y')
        if df is None:
            await ctx.send(f"No data found for {display_name}. Check the symbol and try again.")
            return
        buf = make_chart(df, display_name, '1W', display_count=52, source='yfinance')
        if buf is None:
            await ctx.send(f"Could not generate chart for {display_name}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{display_name}_crypto_weekly.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='cch')
async def crypto_hourly(ctx, symbol: str = 'BTC'):
    """Hourly crypto chart (5 days)."""
    try:
        ticker = resolve_crypto_symbol(symbol)
        display_name = ticker.replace('-USD', '')
        await ctx.send(f"Generating hourly chart for {display_name}...")
        df = get_crypto_bars(symbol, '1h', period='5d')
        if df is None:
            await ctx.send(f"No data found for {display_name}. Check the symbol and try again.")
            return
        buf = make_chart(df, display_name, '1H', display_count=60, source='yfinance')
        if buf is None:
            await ctx.send(f"Could not generate chart for {display_name}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{display_name}_crypto_hourly.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

async def _crypto_chart_minute(ctx, symbol, minutes):
    """Minute-level crypto chart helper."""
    try:
        ticker = resolve_crypto_symbol(symbol)
        display_name = ticker.replace('-USD', '')
        await ctx.send(f"Generating {minutes}min chart for {display_name}...")
        yf_interval = f'{minutes}m' if minutes > 1 else '1m'
        df = get_crypto_bars(symbol, yf_interval, period='1d')
        if df is None or len(df) == 0:
            await ctx.send(f"No data found for {display_name}. Try again later.")
            return
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if len(df) < 2:
            await ctx.send(f"Not enough data for {display_name} yet.")
            return
        buf = make_chart(df, display_name, f'{minutes}m', source='yfinance')
        if buf is None:
            await ctx.send(f"Could not generate chart for {display_name}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{display_name}_crypto_{minutes}min.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='cc1')
async def crypto_1min(ctx, symbol: str = 'BTC'):
    await _crypto_chart_minute(ctx, symbol, 1)

@bot.command(name='cc5')
async def crypto_5min(ctx, symbol: str = 'BTC'):
    await _crypto_chart_minute(ctx, symbol, 5)

@bot.command(name='cc15')
async def crypto_15min(ctx, symbol: str = 'BTC'):
    await _crypto_chart_minute(ctx, symbol, 15)

@bot.command(name='cc30')
async def crypto_30min(ctx, symbol: str = 'BTC'):
    await _crypto_chart_minute(ctx, symbol, 30)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

bot.run(os.getenv('DISCORD_TOKEN'))
