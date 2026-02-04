import discord
from discord.ext import commands
from discord import app_commands
import yfinance as yf
import mplfinance as mpf
import io
import os

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

def create_chart(symbol, period='1y', interval='1wk'):
      ticker = yf.Ticker(symbol)
      df = ticker.history(period=period, interval=interval)
      if df.empty:
                return None, "No data found"
            df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    mc = mpf.make_marketcolors(up='#00ff00', down='#ff00ff', edge='inherit', wick='inherit', volume={'up': '#00ff00', 'down': '#ff00ff'})
    s = mpf.make_mpf_style(marketcolors=mc, base_mpl_style='dark_background', facecolor='#0d1117', figcolor='#0d1117')
    add_plots = []
    add_plots.append(mpf.make_addplot(df['SMA20'], color='#00bfff', width=1))
    add_plots.append(mpf.make_addplot(df['SMA50'], color='#ff1493', width=1))
    buf = io.BytesIO()
    current = df.iloc[-1]
    title = f"{symbol.upper()} | C:{current['Close']:.2f}"
    mpf.plot(df, type='candle', style=s, volume=True, addplot=add_plots, title=title, savefig=dict(fname=buf, dpi=150, bbox_inches='tight'))
    buf.seek(0)
    return buf, None

@bot.command(name='chart')
async def chart_command(ctx, symbol: str = None):
      if not symbol:
                await ctx.send("Usage: !chart SYMBOL (e.g., !chart AAPL)")
                return
            await ctx.send(f"Generating chart for {symbol.upper()}...")
    buf, error = create_chart(symbol)
    if error:
              await ctx.send(f"Error: {error}")
              return
          file = discord.File(buf, filename=f"{symbol}_chart.png")
    await ctx.send(file=file)

@bot.tree.command(name="chart", description="Generate a stock chart")
@app_commands.describe(symbol="Stock ticker symbol (e.g., AAPL)")
async def slash_chart(interaction: discord.Interaction, symbol: str):
      await interaction.response.defer()
    buf, error = create_chart(symbol)
    if error:
              await interaction.followup.send(f"Error: {error}")
              return
          file = discord.File(buf, filename=f"{symbol}_chart.png")
    await interaction.followup.send(file=file)

@bot.event
async def on_ready():
      print(f'{bot.user} is online!')
    await bot.tree.sync()

bot.run(os.getenv('DISCORD_TOKEN'))
