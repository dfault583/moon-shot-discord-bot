import discord
from discord.ext import commands
import yfinance as yf
import mplfinance as mpf
import io
import os

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.command(name='chart')
async def chart(ctx, symbol: str = 'AAPL'):
          await ctx.send(f"Generating chart for {symbol}...")
          ticker = yf.Ticker(symbol)
          df = ticker.history(period='1y', interval='1wk')
          df['SMA20'] = df['Close'].rolling(window=20).mean()
          df['SMA50'] = df['Close'].rolling(window=50).mean()
          mc = mpf.make_marketcolors(up='#00ff00', down='#ff00ff', edge='inherit', wick='inherit', volume={'up': '#00ff00', 'down': '#ff00ff'})
          s = mpf.make_mpf_style(marketcolors=mc, base_mpl_style='dark_background', facecolor='#0d1117', figcolor='#0d1117')
          plots = [mpf.make_addplot(df['SMA20'], color='#00bfff', width=1), mpf.make_addplot(df['SMA50'], color='#ff1493', width=1)]
          buf = io.BytesIO()
          mpf.plot(df, type='candle', style=s, volume=True, addplot=plots, savefig=dict(fname=buf, dpi=150, bbox_inches='tight'))
          buf.seek(0)
          await ctx.send(file=discord.File(buf, filename=f"{symbol}_chart.png"))

@bot.event
async def on_ready():
          print(f'{bot.user} is online!')

bot.run(os.getenv('DISCORD_TOKEN'))
