import discord
from discord.ext import commands
from discord import app_commands
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import io
import os
from datetime import datetime, timedelta

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

def create_chart(symbol: str, period: str = '1y', interval: str = '1wk'):
      """Create a candlestick chart with volume and moving averages"""
      try:
                # Fetch stock data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)

        if df.empty:
                      return None, f"No data found for {symbol}"

        # Calculate moving averages
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()

        # Create custom market colors (green/red candles)
        mc = mpf.make_marketcolors(
                      up='#00ff00',
                      down='#ff00ff',
                      edge='inherit',
                      wick='inherit',
                      volume={'up': '#00ff00', 'down': '#ff00ff'}
        )

        # Create custom style with dark background
        s = mpf.make_mpf_style(
                      marketcolors=mc,
                      base_mpl_style='dark_background',
                      gridstyle='-',
                      gridcolor='#333333',
                      facecolor='#0d1117',
                      figcolor='#0d1117',
                      rc={'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'}
        )

        # Add moving average plots
        add_plots = []
        if not df['SMA20'].isna().all():
                      add_plots.append(mpf.make_addplot(df['SMA20'], color='#00bfff', width=1))
                  if not df['SMA50'].isna().all():
                                add_plots.append(mpf.make_addplot(df['SMA50'], color='#ff1493', width=1))
                            if not df['SMA200'].isna().all():
                                          add_plots.append(mpf.make_addplot(df['SMA200'], color='#ffd700', width=1))

        # Create figure
        buf = io.BytesIO()

        # Get current price info
        current = df.iloc[-1]
        prev_close = df.iloc[-2]['Close'] if len(df) > 1 else current['Close']
        change = current['Close'] - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

        title = f"{symbol.upper()} | O:{current['Open']:.2f} H:{current['High']:.2f} L:{current['Low']:.2f} C:{current['Close']:.2f}"

        fig, axes = mpf.plot(
                      df,
                      type='candle',
                      style=s,
                      volume=True,
                      addplot=add_plots if add_plots else None,
                      title=title,
                      ylabel='Price',
                      ylabel_lower='Volume',
                      figsize=(12, 8),
                      returnfig=True
        )

        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0d1117')
        buf.seek(0)

        return buf, None

except Exception as e:
        return None, str(e)

@bot.event
async def on_ready():
      print(f'{bot.user} has connected to Discord!')
    try:
              synced = await bot.tree.sync()
              print(f"Synced {len(synced)} command(s)")
except Exception as e:
        print(f"Failed to sync commands: {e}")

@bot.tree.command(name="chart", description="Get a stock chart with candlesticks, volume, and moving averages")
@app_commands.describe(
      symbol="Stock ticker symbol (e.g., AAPL, MSFT, TSLA)",
      period="Time period (1mo, 3mo, 6mo, 1y, 2y, 5y)",
      interval="Candle interval (1d, 1wk, 1mo)"
)
async def chart(interaction: discord.Interaction, symbol: str, period: str = "1y", interval: str = "1wk"):
      await interaction.response.defer()

    chart_buf, error = create_chart(symbol.upper(), period, interval)

    if error:
              await interaction.followup.send(f"Error: {error}")
              return

    file = discord.File(chart_buf, filename=f"{symbol.upper()}_chart.png")
    await interaction.followup.send(file=file)

@bot.command(name='chart')
async def chart_prefix(ctx, symbol: str, period: str = "1y", interval: str = "1wk"):
      """Get a stock chart - Usage: !chart AAPL 1y 1wk"""
    async with ctx.typing():
              chart_buf, error = create_chart(symbol.upper(), period, interval)

        if error:
                      await ctx.send(f"Error: {error}")
                      return

        file = discord.File(chart_buf, filename=f"{symbol.upper()}_chart.png")
        await ctx.send(file=file)

# Run the bot
if __name__ == "__main__":
      token = os.getenv('DISCORD_TOKEN')
    if not token:
              print("Error: DISCORD_TOKEN environment variable not set")
else:
        bot.run(token)
