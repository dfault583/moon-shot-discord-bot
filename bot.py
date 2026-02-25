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
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
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

# Futures aliases for yfinance
FUTURES_ALIASES = {
    'ES': 'ES=F', 'SPX': 'ES=F', 'SP500': 'ES=F', 'SPOOS': 'ES=F',
    'NQ': 'NQ=F', 'NASDAQ': 'NQ=F', 'NDX': 'NQ=F',
    'YM': 'YM=F', 'DOW': 'YM=F', 'DJIA': 'YM=F',
    'RTY': 'RTY=F', 'RUSSELL': 'RTY=F', 'RUT': 'RTY=F',
    'CL': 'CL=F', 'OIL': 'CL=F', 'CRUDE': 'CL=F', 'WTI': 'CL=F',
    'GC': 'GC=F', 'GOLD': 'GC=F',
    'SI': 'SI=F', 'SILVER': 'SI=F',
    'HG': 'HG=F', 'COPPER': 'HG=F',
    'NG': 'NG=F', 'NATGAS': 'NG=F',
    'ZB': 'ZB=F', 'BONDS': 'ZB=F', 'TBOND': 'ZB=F',
    'ZN': 'ZN=F', 'TNOTE': 'ZN=F',
    'ZC': 'ZC=F', 'CORN': 'ZC=F',
    'ZS': 'ZS=F', 'SOYBEAN': 'ZS=F',
    'ZW': 'ZW=F', 'WHEAT': 'ZW=F',
    'DX': 'DX=F', 'DOLLAR': 'DX=F', 'DXY': 'DX-Y.NYB',
    '6E': '6E=F', 'EURO': '6E=F', 'EURUSD': '6E=F',
    '6J': '6J=F', 'YEN': '6J=F',
    '6B': '6B=F', 'POUND': '6B=F', 'GBP': '6B=F',
    'VIX': '^VIX', 'VX': 'VX=F',
}

FUTURES_DISPLAY_NAMES = {
    'ES=F': 'ES (S&P 500)', 'NQ=F': 'NQ (Nasdaq 100)', 'YM=F': 'YM (Dow)',
    'RTY=F': 'RTY (Russell 2000)', 'CL=F': 'CL (Crude Oil)', 'GC=F': 'GC (Gold)',
    'SI=F': 'SI (Silver)', 'HG=F': 'HG (Copper)', 'NG=F': 'NG (Nat Gas)',
    'ZB=F': 'ZB (T-Bond)', 'ZN=F': 'ZN (10Y Note)', 'ZC=F': 'ZC (Corn)',
    'ZS=F': 'ZS (Soybean)', 'ZW=F': 'ZW (Wheat)', 'DX=F': 'DX (Dollar Index)',
    'DX-Y.NYB': 'DXY (Dollar Index)', '6E=F': '6E (Euro)', '6J=F': '6J (Yen)',
    '6B=F': '6B (British Pound)', '^VIX': 'VIX', 'VX=F': 'VX (VIX Futures)',
}

def resolve_futures_symbol(symbol):
    """Convert user input to yfinance futures ticker format."""
    symbol = symbol.upper().strip()
    if symbol in FUTURES_ALIASES:
        return FUTURES_ALIASES[symbol]
    if symbol.endswith('=F') or symbol.startswith('^'):
        return symbol
    return f'{symbol}=F'

def get_futures_bars(symbol, yf_interval, period=None, start_date=None, end_date=None):
    """Get futures data from yfinance."""
    ticker_symbol = resolve_futures_symbol(symbol)
    df = get_bars_yfinance(ticker_symbol, yf_interval, period=period, start_date=start_date, end_date=end_date)
    if df is not None and len(df) > 0:
        return df, ticker_symbol
    return None, ticker_symbol


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
        return df, 'yfinance'    return None, None

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

    # Try as futures
    if symbol in FUTURES_ALIASES:
        futures_ticker_str = FUTURES_ALIASES[symbol]
        try:
            ticker = yf.Ticker(futures_ticker_str)
            info = ticker.fast_info
            if info and hasattr(info, 'last_price') and info.last_price is not None:
                result = _build_price_embed(ticker, futures_ticker_str, is_crypto=False)
                if result:
                    return result
        except Exception as e:
            print(f"Futures lookup failed for {futures_ticker_str}: {e}")

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

        embed.set_footer(text=f'{asset_type} • {market_status}')
        return embed
    except Exception as e:
        print(f"Build price embed error: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_vwap(df, band_mult=1.0):
    """Calculate VWAP with standard deviation bands, session-anchored."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    # Detect session boundaries (new day)
    dates = df.index.date if hasattr(df.index, 'date') else pd.Series(df.index).dt.date.values
    vwap = pd.Series(np.nan, index=df.index)
    upper_band = pd.Series(np.nan, index=df.index)
    lower_band = pd.Series(np.nan, index=df.index)
    cum_vol = 0.0
    cum_tp_vol = 0.0
    cum_tp2_vol = 0.0
    prev_date = None
    for i in range(len(df)):
        cur_date = dates[i]
        if prev_date is None or cur_date != prev_date:
            cum_vol = 0.0
            cum_tp_vol = 0.0
            cum_tp2_vol = 0.0
        v = df['volume'].iloc[i]
        t = tp.iloc[i]
        cum_vol += v
        cum_tp_vol += t * v
        cum_tp2_vol += t * t * v        if cum_vol > 0:
            vwap_val = cum_tp_vol / cum_vol
            vwap.iloc[i] = vwap_val
            variance = (cum_tp2_vol / cum_vol) - (vwap_val * vwap_val)
            stdev = np.sqrt(max(variance, 0))
            upper_band.iloc[i] = vwap_val + band_mult * stdev
            lower_band.iloc[i] = vwap_val - band_mult * stdev
        prev_date = cur_date
    return vwap, upper_band, lower_band


def calculate_support_resistance(df, left_bars=15, right_bars=15, volume_thresh=20):
    """Calculate Support and Resistance levels with breaks (LuxAlgo style).
    Uses pivot highs/lows to find S/R levels, marks breaks confirmed by volume.
    """
    n = len(df)
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values
    volumes = df['volume'].values

    # Find pivot highs and pivot lows
    pivot_high_vals = np.full(n, np.nan)
    pivot_low_vals = np.full(n, np.nan)

    for i in range(left_bars, n - right_bars):
        # Pivot high: highest high in window
        is_ph = True
        for j in range(i - left_bars, i + right_bars + 1):
            if j != i and highs[j] >= highs[i]:
                is_ph = False
                break
        if is_ph:
            pivot_high_vals[i] = highs[i]

        # Pivot low: lowest low in window
        is_pl = True
        for j in range(i - left_bars, i + right_bars + 1):
            if j != i and lows[j] <= lows[i]:
                is_pl = False
                break
        if is_pl:
            pivot_low_vals[i] = lows[i]

    # Forward-fill pivot values (Pine's fixnan equivalent), shifted by 1
    resistance = pd.Series(pivot_high_vals).shift(1).ffill().values
    support = pd.Series(pivot_low_vals).shift(1).ffill().values

    # Volume oscillator: EMA(5) vs EMA(10)
    vol_series = pd.Series(volumes, dtype='float64')
    vol_short = vol_series.ewm(span=5, adjust=False).mean()
    vol_long = vol_series.ewm(span=10, adjust=False).mean()
    vol_osc = (100 * (vol_short - vol_long) / vol_long).values

    # Detect breaks
    break_down = []  # support breaks
    break_up = []    # resistance breaks
    bull_wick = []
    bear_wick = []

    for i in range(1, n):
        if np.isnan(support[i]) or np.isnan(resistance[i]):
            continue
        # Break below support with volume
        if closes[i] < support[i] and closes[i - 1] >= support[i]:
            is_bear_wick = (opens[i] - closes[i]) < (highs[i] - opens[i])
            if is_bear_wick:
                bear_wick.append(i)
            elif vol_osc[i] > volume_thresh:
                break_down.append(i)
        # Break above resistance with volume
        if closes[i] > resistance[i] and closes[i - 1] <= resistance[i]:
            is_bull_wick = (opens[i] - lows[i]) > (closes[i] - opens[i])
            if is_bull_wick:
                bull_wick.append(i)
            elif vol_osc[i] > volume_thresh:
                break_up.append(i)

    return {
        'resistance': resistance,
        'support': support,
        'break_down': break_down,
        'break_up': break_up,
        'bull_wick': bull_wick,
        'bear_wick': bear_wick,
    }


def calculate_volume_profile(df, num_bins=100):
    """Calculate volume profile: volume distribution at each price level."""
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min
    if price_range <= 0:
        return None
    bin_size = price_range / num_bins
    bins = np.zeros(num_bins)
    for i in range(len(df)):
        bar_low = df['low'].iloc[i]
        bar_high = df['high'].iloc[i]
        bar_vol = df['volume'].iloc[i]
        is_up = df['close'].iloc[i] >= df['open'].iloc[i]
        low_bin = max(0, int((bar_low - price_min) / bin_size))
        high_bin = min(num_bins - 1, int((bar_high - price_min) / bin_size))
        num_touched = high_bin - low_bin + 1
        if num_touched > 0:
            vol_per_bin = bar_vol / num_touched
            for b in range(low_bin, high_bin + 1):
                bins[b] += vol_per_bin
    # Calculate value area (68% of total volume around POC)
    poc_idx = np.argmax(bins)
    total_vol = bins.sum()
    va_target = total_vol * 0.68
    va_sum = bins[poc_idx]
    va_low_idx = poc_idx
    va_high_idx = poc_idx
    while va_sum < va_target:
        up_vol = bins[va_high_idx + 1] if va_high_idx < num_bins - 1 else 0
        dn_vol = bins[va_low_idx - 1] if va_low_idx > 0 else 0
        if up_vol == 0 and dn_vol == 0:
            break
        if up_vol >= dn_vol:
            va_high_idx += 1
            va_sum += up_vol
        else:
            va_low_idx -= 1
            va_sum += dn_vol
    price_levels = np.array([price_min + (i + 0.5) * bin_size for i in range(num_bins)])
    poc_price = price_levels[poc_idx]
    vah_price = price_levels[va_high_idx]
    val_price = price_levels[va_low_idx]
    return {
        'bins': bins,
        'price_levels': price_levels,
        'poc_idx': poc_idx,
        'poc_price': poc_price,
        'vah_price': vah_price,
        'val_price': val_price,
        'va_low_idx': va_low_idx,
        'va_high_idx': va_high_idx,
        'max_vol': bins.max(),
    }

def make_chart(df, symbol, timeframe, display_count=None, source=None):
    try:
        if len(df) < 2:
            print(f"Not enough data for {symbol} {timeframe}: {len(df)} rows")
            return None

        # Calculate indicators on FULL data first
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        df['SMA200'] = df['close'].rolling(window=200).mean()
        # Only calculate VWAP for intraday minute charts
        if timeframe in ('1m', '5m', '15m', '30m'):
            df['VWAP'], df['VWAP_UPPER'], df['VWAP_LOWER'] = calculate_vwap(df, band_mult=1.0)
        
        # Trim to display window
        if display_count and len(df) > display_count:
            df = df.iloc[-display_count:]

        # === Main chart addplots ===
        plots = []
        legend_items = []

        if df['SMA20'].notna().any():
            plots.append(mpf.make_addplot(df['SMA20'], color='#00bcd4', width=1.0, panel=0))
            legend_items.append(('SMA 20', '#00bcd4', '-'))
        if df['SMA50'].notna().any():
            plots.append(mpf.make_addplot(df['SMA50'], color='#ff6d00', width=1.0, panel=0))
            legend_items.append(('SMA 50', '#ff6d00', '-'))
        if df['SMA200'].notna().any():
            plots.append(mpf.make_addplot(df['SMA200'], color='#ab47bc', width=1.2, panel=0))
            legend_items.append(('SMA 200', '#ab47bc', '-'))
        if 'VWAP' in df.columns and df['VWAP'].notna().any():
            plots.append(mpf.make_addplot(df['VWAP'], color='#ffeb3b', width=1.2, panel=0))
            legend_items.append(('VWAP', '#ffeb3b', '-'))

        # Buy/Sell markers on main chart
        # === Chart Style ===
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
                'font.size': 11,
                'axes.labelcolor': TV_TEXT,                'axes.edgecolor': TV_BORDER,
                'xtick.color': TV_TEXT,
                'ytick.color': TV_TEXT,
                'ytick.labelsize': 12,
                'xtick.labelsize': 10,
                'text.color': TV_TEXT,
                'figure.titlesize': 13,
                'axes.titlesize': 13,
            }
        )

        last_close = df['close'].iloc[-1]
        # For intraday charts, compare to day's open; for daily+ compare to previous close
        if timeframe in ('1m', '5m', '15m', '30m', '1H'):
            ref_price = df['open'].iloc[0]
        else:
            ref_price = df['close'].iloc[-2] if len(df) > 1 else last_close
        change = last_close - ref_price
        pct_change = (change / ref_price) * 100
        sign = '+' if change >= 0 else ''
        src_tag = ' [YF]' if source == 'yfinance' else ''
        title = f'{symbol} {timeframe}  {last_close:.2f}  {sign}{change:.2f} ({sign}{pct_change:.2f}%){src_tag}'

        buf = io.BytesIO()

        fig, axes = mpf.plot(
            df, type='candle', style=s, volume=True,
            addplot=plots if plots else None,
            figsize=(14, 9), tight_layout=False,
            scale_padding={'left': 0.05, 'top': 1.2, 'right': 0.5, 'bottom': 0.8},
            returnfig=True,
            volume_panel=1,
            panel_ratios=(4, 1)
        )

        fig.suptitle(title, color='#ffffff', fontsize=14, fontweight='bold', x=0.08, ha='left')

        # === TradingView-style Watermark ===
        price_ax = axes[0]
        # Large ticker + timeframe text
        watermark_text = f'{symbol}, {timeframe}'
        price_ax.text(
            0.5, 0.5, watermark_text,
            transform=price_ax.transAxes,
            fontsize=52, fontweight='bold',
            color='#4a4e69', alpha=0.25,
            ha='center', va='center',
            zorder=0
        )
        # Company/asset name below (try to look up via yfinance)
        try:
            ticker_info = yf.Ticker(symbol).info
            company_name = ticker_info.get('shortName') or ticker_info.get('longName', '')
            if company_name:
                price_ax.text(
                    0.5, 0.42, company_name,
                    transform=price_ax.transAxes,
                    fontsize=24, fontweight='normal',
                    color='#4a4e69', alpha=0.20,
                    ha='center', va='center',
                    zorder=0
                )
        except Exception:
            pass  # Skip company name if lookup fails

        # Add indicator legend
        if legend_items:
            handles = []
            for name, color, ls in legend_items:
                handles.append(Line2D([0], [0], color=color, linewidth=1.5, linestyle=ls, label=name))
            axes[0].legend(
                handles=handles,
                loc='upper left',
                fontsize=7,
                facecolor=TV_BG,
                edgecolor=TV_BORDER,
                labelcolor=TV_TEXT,
                framealpha=0.8,
                borderpad=0.4,
                handlelength=1.5
            )

        for ax in axes:
            ax.set_facecolor(TV_BG)
            ax.tick_params(colors=TV_TEXT, labelsize=11)
            # Make y-axis price labels larger and bolder
            ax.yaxis.set_tick_params(labelsize=12)
            for label in ax.yaxis.get_ticklabels():
                label.set_fontweight('bold')
                label.set_color('#ffffff')
            for spine in ax.spines.values():
                spine.set_color(TV_BORDER)

# === Volume Profile (overlay using blended transform) ===
        vp = calculate_volume_profile(df, num_bins=80)
        if vp is not None:
            from matplotlib.transforms import blended_transform_factory
            price_ax = axes[0]
            max_vol = vp['max_vol']
            # Use blended transform: x in axes coords (0-1), y in data coords (price)
            trans = blended_transform_factory(price_ax.transAxes, price_ax.transData)
            bin_height = vp['price_levels'][1] - vp['price_levels'][0] if len(vp['price_levels']) > 1 else 0.01
            for i in range(len(vp['bins'])):
                vol = vp['bins'][i]
                if vol <= 0:
                    continue
                bar_width_pct = (vol / max_vol) * 0.20  # max 20% of axes width
                y = vp['price_levels'][i]
                if vp['va_low_idx'] <= i <= vp['va_high_idx']:
                    bar_color = (0.2, 0.4, 0.8, 0.35)
                else:
                    bar_color = (0.5, 0.5, 0.5, 0.25)
                # Draw from right edge inward: left = 1.0 - bar_width, width = bar_width
                price_ax.barh(y, bar_width_pct, height=bin_height * 0.9, left=1.0 - bar_width_pct, color=bar_color, transform=trans, zorder=1, clip_on=True)
            # POC line
            price_ax.axhline(y=vp['poc_price'], color='#ff0000', linewidth=0.8, linestyle='-', alpha=0.7, zorder=2)
            # VAH/VAL lines
            price_ax.axhline(y=vp['vah_price'], color='#2962ff', linewidth=0.6, linestyle='--', alpha=0.5, zorder=2)
            price_ax.axhline(y=vp['val_price'], color='#2962ff', linewidth=0.6, linestyle='--', alpha=0.5, zorder=2)
            legend_items.append(('POC', '#ff0000', '-'))
            legend_items.append(('VAH/VAL', '#2962ff', '--'))
            # Re-draw legend with VP entries
            handles = []
            for name, color, ls in legend_items:
                handles.append(Line2D([0], [0], color=color, linewidth=1.5, linestyle=ls, label=name))
            price_ax.legend(
                handles=handles,
                loc='upper left',
                fontsize=7,
                facecolor=TV_BG,
                edgecolor=TV_BORDER,
                labelcolor=TV_TEXT,
                framealpha=0.8,
                borderpad=0.4,
                handlelength=1.5
            )

        # === Support & Resistance Levels with Breaks ===
        try:
            sr = calculate_support_resistance(df, left_bars=15, right_bars=15, volume_thresh=20)
            price_ax = axes[0]
            n = len(df)
            offset = 15 + 1  # rightBars + 1, matching Pine offset

            # Draw resistance line (red)
            res_series = sr['resistance']
            for i in range(offset + 1, n):
                if not np.isnan(res_series[i]) and not np.isnan(res_series[i - 1]):
                    if res_series[i] == res_series[i - 1]:
                        price_ax.plot([i - offset - 1, i - offset], [res_series[i - 1], res_series[i]],
                                      color='#FF0000', linewidth=2, alpha=0.85, zorder=4)

            # Draw support line (blue)
            sup_series = sr['support']
            for i in range(offset + 1, n):
                if not np.isnan(sup_series[i]) and not np.isnan(sup_series[i - 1]):
                    if sup_series[i] == sup_series[i - 1]:
                        price_ax.plot([i - offset - 1, i - offset], [sup_series[i - 1], sup_series[i]],
                                      color='#233dee', linewidth=2, alpha=0.85, zorder=4)

            # Break markers
            for idx in sr['break_down']:
                price_ax.annotate('B', xy=(idx, df['high'].iloc[idx]),
                                  fontsize=7, fontweight='bold', color='white',
                                  ha='center', va='bottom',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='#ff1744', ec='none', alpha=0.9),
                                  zorder=6)
            for idx in sr['break_up']:
                price_ax.annotate('B', xy=(idx, df['low'].iloc[idx]),
                                  fontsize=7, fontweight='bold', color='white',
                                  ha='center', va='top',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='#00e676', ec='none', alpha=0.9),
                                  zorder=6)
            for idx in sr['bear_wick']:
                price_ax.annotate('BW', xy=(idx, df['high'].iloc[idx]),
                                  fontsize=6, fontweight='bold', color='white',
                                  ha='center', va='bottom',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='#ff1744', ec='none', alpha=0.9),
                                  zorder=6)
            for idx in sr['bull_wick']:
                price_ax.annotate('BW', xy=(idx, df['low'].iloc[idx]),
                                  fontsize=6, fontweight='bold', color='white',
                                  ha='center', va='top',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='#00e676', ec='none', alpha=0.9),
                                  zorder=6)
        except Exception as e:
            print(f"S/R error: {e}")

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
            '**!cm SYMBOL** — Monthly chart (2 years)
'
            '**!cw SYMBOL** — Weekly chart (1 year)
'
            '**!cd SYMBOL** — Daily chart (3 months)
'
            '**!ch SYMBOL** — Hourly chart (5 days)
'
            '**!c1m SYMBOL** — 1 min chart (today)
'
            '**!c5m SYMBOL** — 5 min chart (today)
'
            '**!c15m SYMBOL** — 15 min chart (today)
'
            '**!c30m SYMBOL** — 30 min chart (today)'
        ),
        inline=False
    )
    embed.add_field(
        name='Crypto Charts',
        value=(
            '**!ccm SYMBOL** — Monthly chart (2 years)
'
            '**!ccw SYMBOL** — Weekly chart (1 year)
'
            '**!ccd SYMBOL** — Daily chart (3 months)
'
            '**!cch SYMBOL** — Hourly chart (5 days)
'
            '**!cc1m SYMBOL** — 1 min chart (today)
'
            '**!cc5m SYMBOL** — 5 min chart (today)
'
            '**!cc15m SYMBOL** — 15 min chart (today)
'
            '**!cc30m SYMBOL** — 30 min chart (today)
'
            '
Examples: !cc BTC, !ccw ETH, !cch SOL'
        ),
        inline=False
    )
    embed.add_field(
        name='Futures Charts',
        value=(
            '**!fm SYMBOL** — Monthly chart (2 years)
'
            '**!fw SYMBOL** — Weekly chart (1 year)
'
            '**!fd SYMBOL** — Daily chart (3 months)
'
            '**!fh SYMBOL** — Hourly chart (5 days)
'
            '**!f1m SYMBOL** — 1 min chart (today)
'
            '**!f5m SYMBOL** — 5 min chart (today)
'
            '**!f15m SYMBOL** — 15 min chart (today)
'
            '**!f30m SYMBOL** — 30 min chart (today)
'
            '
Symbols: ES, NQ, YM, RTY, CL, GC, SI, NG, VIX, DXY, ZB, 6E & more
'
            'Examples: !fd ES, !fh NQ, !f5m GOLD'
        ),
        inline=False
    )
    embed.add_field(
        name='Price Check',
        value=(
            '**!p / !price SYMBOL** — Quick price check (stocks & crypto)
'
            'Shows current price, change, pre-market/after-hours data
'
            'Example: !p AAPL, !p BTC, !p TSLA'
        ),
        inline=False
    )
    embed.add_field(
        name='Overlays',
        value='SMA 20 / 50 / 200 + VWAP (minute charts only) + Support & Resistance + Volume Profile (POC, Value Area)',
        inline=False
    )
    embed.add_field(
        name='Data',
        value='Stocks: NYSE, NASDAQ & OTC (Alpaca + Yahoo Finance)
Crypto: BTC, ETH, SOL, XRP, DOGE, ADA, and 30+ more via Yahoo Finance',
        inline=False
    )
    embed.add_field(
        name='Options',
        value='**!10bagger** — Current high-risk/high-reward option trade pick
**!retardspecial** — The most degenerate trade imaginable
**!JOINIS** — Top 3 highest probability call plays (Unusual Whales flow)',
        inline=False
    )
    embed.set_footer(text='Default stock: AAPL | Default crypto: BTC')
    await ctx.send(embed=embed)

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
        start_date = end_date - timedelta(days=365*5)
        df, source = get_bars(symbol, TimeFrame.Week, '1wk', start_date, end_date, yf_period='5y')
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
        start_date = end_date - timedelta(days=730)
        df, source = get_bars(symbol, TimeFrame.Day, '1d', start_date, end_date, yf_period='2y')
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
        start_date = end_date - timedelta(days=90)
        df, source = get_bars(symbol, TimeFrame.Hour, '1h', start_date, end_date, yf_period='1mo')
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

@bot.command(name='c1m')
async def chart_1min(ctx, symbol: str = 'AAPL'):
    await _chart_minute(ctx, symbol, 1)

@bot.command(name='c5m')
async def chart_5min(ctx, symbol: str = 'AAPL'):
    await _chart_minute(ctx, symbol, 5)

@bot.command(name='c15m')
async def chart_15min(ctx, symbol: str = 'AAPL'):
    await _chart_minute(ctx, symbol, 15)

@bot.command(name='c30m')
async def chart_30min(ctx, symbol: str = 'AAPL'):
    await _chart_minute(ctx, symbol, 30)


# ============================================================
# PRICE COMMAND
# ============================================================

@bot.command(name='p', aliases=['price'])
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
        df = get_crypto_bars(symbol, '1d', period='2y')
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
        df = get_crypto_bars(symbol, '1wk', period='5y')
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
        df = get_crypto_bars(symbol, '1h', period='1mo')
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

@bot.command(name='cc1m')
async def crypto_1min(ctx, symbol: str = 'BTC'):
    await _crypto_chart_minute(ctx, symbol, 1)

@bot.command(name='cc5m')
async def crypto_5min(ctx, symbol: str = 'BTC'):
    await _crypto_chart_minute(ctx, symbol, 5)

@bot.command(name='cc15m')
async def crypto_15min(ctx, symbol: str = 'BTC'):
    await _crypto_chart_minute(ctx, symbol, 15)

@bot.command(name='cc30m')
async def crypto_30min(ctx, symbol: str = 'BTC'):
    await _crypto_chart_minute(ctx, symbol, 30)

# ============================================================
# FUTURES COMMANDS
# ============================================================

@bot.command(name='fd')
async def futures_daily(ctx, symbol: str = 'ES'):
    """Daily futures chart (3 months)."""
    try:
        df, ticker = get_futures_bars(symbol, '1d', period='2y')
        display_name = FUTURES_DISPLAY_NAMES.get(ticker, ticker.replace('=F', ''))
        if df is None:
            await ctx.send(f"No data found for {display_name}. Check the symbol and try again.")
            return
        await ctx.send(f"Generating daily chart for {display_name}...")
        buf = make_chart(df, display_name, '1D', display_count=65, source='yfinance')
        if buf is None:
            await ctx.send(f"Could not generate chart for {display_name}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_futures_daily.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='fw')
async def futures_weekly(ctx, symbol: str = 'ES'):
    """Weekly futures chart (1 year)."""
    try:
        df, ticker = get_futures_bars(symbol, '1wk', period='5y')
        display_name = FUTURES_DISPLAY_NAMES.get(ticker, ticker.replace('=F', ''))
        if df is None:
            await ctx.send(f"No data found for {display_name}. Check the symbol and try again.")
            return
        await ctx.send(f"Generating weekly chart for {display_name}...")
        buf = make_chart(df, display_name, '1W', display_count=52, source='yfinance')
        if buf is None:
            await ctx.send(f"Could not generate chart for {display_name}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_futures_weekly.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='fm')
async def futures_monthly(ctx, symbol: str = 'ES'):
    """Monthly futures chart (2 years)."""
    try:
        df, ticker = get_futures_bars(symbol, '1mo', period='5y')
        display_name = FUTURES_DISPLAY_NAMES.get(ticker, ticker.replace('=F', ''))
        if df is None:
            await ctx.send(f"No data found for {display_name}. Check the symbol and try again.")
            return
        await ctx.send(f"Generating monthly chart for {display_name}...")
        buf = make_chart(df, display_name, '1M', source='yfinance')
        if buf is None:
            await ctx.send(f"Could not generate chart for {display_name}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_futures_monthly.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='fh')
async def futures_hourly(ctx, symbol: str = 'ES'):
    """Hourly futures chart (5 days)."""
    try:
        df, ticker = get_futures_bars(symbol, '1h', period='1mo')
        display_name = FUTURES_DISPLAY_NAMES.get(ticker, ticker.replace('=F', ''))
        if df is None:
            await ctx.send(f"No data found for {display_name}. Check the symbol and try again.")
            return
        await ctx.send(f"Generating hourly chart for {display_name}...")
        buf = make_chart(df, display_name, '1H', display_count=60, source='yfinance')
        if buf is None:
            await ctx.send(f"Could not generate chart for {display_name}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_futures_hourly.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

async def _futures_chart_minute(ctx, symbol, minutes):
    """Minute-level futures chart helper."""
    try:
        yf_interval = f'{minutes}m' if minutes > 1 else '1m'
        df, ticker = get_futures_bars(symbol, yf_interval, period='1d')
        display_name = FUTURES_DISPLAY_NAMES.get(ticker, ticker.replace('=F', ''))
        if df is None or len(df) == 0:
            await ctx.send(f"No data found for {display_name}. Try again later.")
            return
        await ctx.send(f"Generating {minutes}min chart for {display_name}...")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if len(df) < 2:
            await ctx.send(f"Not enough data for {display_name} yet.")
            return
        buf = make_chart(df, display_name, f'{minutes}m', source='yfinance')
        if buf is None:
            await ctx.send(f"Could not generate chart for {display_name}.")
            return
        await ctx.send(file=discord.File(buf, filename=f'{symbol}_futures_{minutes}min.png'))
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='f1m')
async def futures_1min(ctx, symbol: str = 'ES'):
    await _futures_chart_minute(ctx, symbol, 1)

@bot.command(name='f5m')
async def futures_5min(ctx, symbol: str = 'ES'):
    await _futures_chart_minute(ctx, symbol, 5)

@bot.command(name='f15m')
async def futures_15min(ctx, symbol: str = 'ES'):
    await _futures_chart_minute(ctx, symbol, 15)

@bot.command(name='f30m')
async def futures_30min(ctx, symbol: str = 'ES'):
    await _futures_chart_minute(ctx, symbol, 30)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')


# ============================================================
# 10-BAGGER OPTION TRADE COMMAND
# ============================================================
@bot.command(name='10bagger')
async def ten_bagger(ctx):
    """Show the current high-risk high-reward option trade pick."""
    embed = discord.Embed(
        title='U0001f680 10-Bagger Option Play U0001f680',
        description='High-risk, high-reward option trade pick based on unusual options flow & current market trends. **This is NOT financial advice. Do your own research.**',
        color=0xffeb3b
    )
    embed.add_field(
        name='U0001f4c8 Trade Setup',
        value=(
            '**Ticker:** JBLU (JetBlue Airways)
'
            '**Contract:** $7 Call
'
            '**Expiration:** Jun 18, 2026
'
            '**Entry Price:** ~$0.76
'
            '**Type:** OTM Call (~10.8% out of the money)
'
            '**Cost:** ~$76 per contract'
        ),
        inline=False
    )
    embed.add_field(
        name='U0001f3af Why This Trade',
        value=(
            '• UW Flow: 20,000+ contracts bought across multiple 5K blocks — all on the ask side
'
            '• Call volume 49,548 vs put volume 1,165 — P/C ratio of 0.02 (insanely bullish)
'
            '• $2.7M in call premium today alone — massive institutional positioning
'
            '• Stock up 4%+ today on heavy volume — accumulation pattern
'
            '• 120 DTE gives time through summer travel season — peak revenue catalyst
'
            '• Airline sector turnaround — travel demand at record highs, costs stabilizing'
        ),
        inline=False
    )
    embed.add_field(
        name='U0001f4b0 10-Bagger Math',
        value=(
            '• Entry: ~$0.76/contract ($76 per contract)
'
            '• Breakeven: JBLU hits $7.76 (+22.8%) by Jun 18
'
            '• **10x Target:** JBLU hits ~$14.60 → call worth ~$7.60 (10x) U0001f680
'
            '• **If JBLU hits $12 (52W high area):** call worth ~$5.00 (6.6x) U0001f680
'
            '• **If JBLU hits $15+ (analyst targets):** call worth ~$8.00 (10.5x) U0001f680U0001f680
'
            '• Summer travel + earnings 5/5 could send airlines parabolic'
        ),
        inline=False
    )
    embed.add_field(
        name='U0001f525 News & Trend Context',
        value=(
            '• 52W range $3.34-$7.83 — stock near lows with room to run
'
            '• Summer 2026 bookings trending strong — airlines seeing record passenger loads
'
            '• Fuel costs declining — major margin tailwind for airlines
'
            '• JBLU restructuring plan in motion — cutting unprofitable routes, focusing on premium
'
            '• Multiple whale-size blocks all in same strike/expiry — coordinated smart money bet'
        ),
        inline=False
    )
    embed.add_field(
        name='⚠️ Risk Level',
        value='**EXTREME** — OTM options expire worthless most of the time. JBLU is a turnaround story that could fail. Airlines are cyclical and exposed to fuel costs, macro conditions, and competition. Only play with money you can afford to lose entirely.',
        inline=False
    )
    embed.set_footer(text='Updated: Feb 18, 2026 • Source: UW Flow + Market Trends • Not financial advice • DYOR')
    await ctx.send(embed=embed)

# ============================================================
# RETARD SPECIAL COMMAND
# ============================================================
@bot.command(name='retardspecial')
async def retard_special(ctx):
    """The most degenerate option trade imaginable."""
    embed = discord.Embed(
        title='U0001f921 Retard Special U0001f921',
        description='The absolute most degenerate trade possible. **This is NOT financial advice. This is financial self-harm.**',
        color=0xff0000
    )
    embed.add_field(
        name='U0001f4a5 The Play',
        value=(
            '**Ticker:** IBRX (ImmunityBio)
'
            '**Contract:** $7.50 Call
'
            '**Expiration:** Feb 20, 2026
'
            '**Entry Price:** ~$0.20
'
            '**Type:** OTM Call on a stock already up 33% today
'
            '**Cost:** $20 per contract
'
            '**DTE:** 2 DAYS'
        ),
        inline=False
    )
    embed.add_field(
        name='U0001f4a8 Why This Is Insane',
        value=(
            '• IBRX is up 33% TODAY and you're chasing it with 2-day calls
'
            '• SWEEP order detected on UW — someone hit multiple exchanges simultaneously
'
            '• 95,888 call contracts traded vs 25,048 puts — $10.2M in call premium on a biotech
'
            '• Stock went from $6.06 to $8.12 intraday — you're buying at the top of a 33% rip
'
            '• Earnings on 3/2 with EXPECTED move of 27% — but that's AFTER your calls expire
'
            '• 52W high is $8.28 — stock is pennies from ATH and you want MORE'
        ),
        inline=False
    )
    embed.add_field(
        name='U0001f911 Degen Math',
        value=(
            '• Entry: ~$0.20/contract ($20 per contract)
'
            '• Breakeven: IBRX hits $7.70 by Friday
'
            '• **If IBRX hits $8.50 (new ATH):** $1.00/contract = 5x U0001f680
'
            '• **If IBRX hits $9.00:** $1.50/contract = 7.5x U0001f680
'
            '• **If IBRX hits $10.00 (short squeeze territory):** $2.50/contract = 12.5x U0001f680U0001f680
'
            '• At $20 a contract this is literally cheaper than lunch'
        ),
        inline=False
    )
    embed.add_field(
        name='U0001f525 News & Trend Context',
        value=(
            '• IBRX ripping 33% on heavy volume — possible FDA catalyst or data leak
'
            '• $12.7M total options premium today — massive for a $7.8B biotech
'
            '• Net premium +$1.4M bullish — smart money loading calls aggressively
'
            '• Earnings 3/2 with 27% expected move — but these calls expire BEFORE earnings
'
            '• Biotech momentum is contagious — one positive headline could extend the run'
        ),
        inline=False
    )
    embed.add_field(
        name='☢️ Risk Level',
        value="**BEYOND EXTREME** — You're chasing a 33% mover with 2-day expiration calls on a biotech. The stock already had its move. You're betting it goes ANOTHER 30%+ in 48 hours because some sweep orders told you to. This isn't investing. This isn't gambling. This is setting money on fire and hoping the ashes spell 'profit'.",
        inline=False
    )
    embed.set_footer(text='Updated: Feb 18, 2026 • Source: UW Flow + Market Trends • Not financial advice • Pure degeneracy • DYOR')
    await ctx.send(embed=embed)

# ============================================================
# JOINIS COMMAND - Highest Probability Plays (Unusual Whales Flow)
# ============================================================
@bot.command(name='JOINIS', aliases=['joinis'])
async def joinis(ctx):
    """Top 3 highest probability single-leg call trades from UW flow."""
    embed = discord.Embed(
        title='U0001f3af JOINIS — Top 3 High Probability Calls U0001f3af',
        description='Highest conviction single-leg call plays sourced from Unusual Whales flow data + current market trends. **This is NOT financial advice. Do your own research.**',
        color=0x26a69a
    )
    # Trade 1: NCLH
    embed.add_field(
        name='U0001f7e2 #1 — NCLH (Norwegian Cruise Line) $23.5C 02/20',
        value=(
            '**Entry:** ~$0.51 ($51/contract)
'
            '**DTE:** 8 days • **Stock:** $22.62 • **Only 3.9% OTM**
'
            '**UW Signal:** 4,272 contracts bought on ask • $218K premium • 99% ask
'
            '**Volume:** 4,300 vs OI 403 — 10x the open interest in new buying
'
            '**Thesis:** Travel/leisure sector strong, barely OTM, massive institutional size
'
            '**News:** Consumer spending resilient, cruise bookings at record highs for 2026
'
            '**Breakeven:** NCLH hits $24.01 (+6.1%) by Feb 20'
        ),
        inline=False
    )
    # Trade 2: CART
    embed.add_field(
        name='U0001f7e2 #2 — CART (Maplebear/Instacart) $35C 02/20',
        value=(
            '**Entry:** ~$1.04 ($104/contract)
'
            '**DTE:** 8 days • **Stock:** $33.28 • **Only 5.2% OTM**
'
            '**UW Signal:** 1,000 contracts bought on ask • $104K premium • 100% ask
'
            '**Volume:** 1,000 vs OI 278 — 3.6x open interest, all new
'
            '**Thesis:** E-commerce/delivery momentum, CART near breakout level
'
            '**News:** AI-powered grocery delivery growing, CART expanding ad revenue
'
            '**Breakeven:** CART hits $36.04 (+8.3%) by Feb 20'
        ),
        inline=False
    )
    # Trade 3: MU
    embed.add_field(
        name='U0001f7e2 #3 — MU (Micron Technology) $467.5C 02/20',
        value=(
            '**Entry:** ~$3.00 ($300/contract)
'
            '**DTE:** 8 days • **Stock:** $411.27 • **13.7% OTM**
'
            '**UW Signal:** #1 Net Impact on all of UW — most bullish flow in entire market
'
            '**Volume:** 135 • $34K premium • 95% ask
'
            '**Thesis:** Memory chip demand exploding for AI, MU top net bullish impact across all stocks
'
            '**News:** HBM3E demand from NVDA surging, memory super-cycle narrative, AI buildout accelerating
'
            '**Breakeven:** MU hits $470.50 (+14.4%) by Feb 20'
        ),
        inline=False
    )
    embed.add_field(
        name='U0001f525 Market Trend Context',
        value=(
            '• **Top Bullish Flow (UW Net Impact):** MU, SNDK, NVDA, GOOGL, VRT, CRWV, META
'
            '• **Bearish Flow:** MDB, AVGO, AMD, TSM, MSTR, MSFT, ADBE, TSLA
'
            '• SPY at $694 — market holding near highs despite CPI uncertainty
'
            '• AI/compute and travel/leisure are the strongest sector flows today'
        ),
        inline=False
    )
    embed.add_field(
        name='⚠️ Risk',
        value='All plays are OTM calls that can expire worthless. Only trade with money you can afford to lose. These are based on unusual options flow + market trends, not guaranteed outcomes.',
        inline=False
    )
    embed.set_footer(text='Updated: Feb 12, 2026 • Source: Unusual Whales Flow + Market Trends • Not financial advice • DYOR')
    await ctx.send(embed=embed)

bot.run(os.getenv('DISCORD_TOKEN'))
        v = df['volume'].iloc[i]
        t = tp.iloc[i]
        cum_vol += v
        cum_tp_vol += t * v
        cum_tp2_vol += t * t * v
