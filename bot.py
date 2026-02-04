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
      try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                if df.empty:
                              return None, "No data found"
                          df['SMA20'] = df['Close'].rolling(window=20).mean()
                df['SMA50'] = df['Close'].rolling(window=50).mean()
                mc = mpf.make_marketcolors(up='#00ff00', down='#ff00ff', edge='inherit', wick='inherit', volume={'up': '#00ff00', 'down': '#ff00ff'})
                s = mpf.make_mpf_style(marketcolors=mc, base_mpl_style='dark_background', facecolor='#0d1117', figcolor='#0d1117')
                add_plots = []
                if not df['SMA20'].isna().all():
                              add_plots.append(mpf.make_addplot(df['SMA20'], color='#00bfff', width=1))
                          if not df['SMA50'].isna().all():
                                        add_plots.append(mpf.make_addplot(df['SMA50'], color='#ff1493', width=1))
                                    buf = io.BytesIO()
                current = df.iloc[-1]
                title = f"{symbol.upper()} | C:{current['Close']:.2f}"
                fig, axes = mpf.plot(df, type='candle', style=s, volume=True, addplot=add_plots if add_plots else None, title=title, figsize=(12, 8), returnfig=True)
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0d1117')
                buf.seek(0)
                return buf, None
except Exception as e:
        return None, str(e)

@bot.event
async def on_ready():
      print(f'{bot.user} connected!')
      try:
                synced = await bot.tree.sync()
                print(f"Synced {len(synced)} commands")
except Exception as e:
        print(f"Sync failed: {e}")

@bot.tree.command(name="chart", description="Get stock chart")
@app_commands.describe(symbol="Ticker", period="Period", interval="Interval")
async def chart(interaction: discord.Interaction, symbol: str, period: str = "1y", interval: str = "1wk"):
      await interaction.response.defer()
      buf, err = create_chart(symbol.upper(), period, interval)
      if err:
                await interaction.followup.send(f"Error: {err}")
                return
            await interaction.followup.send(file=discord.File(buf, filename=f"{symbol}_chart.png"))

@bot.command(name='chart')
async def chart_cmd(ctx, symbol: str, period: str = "1y", interval: str = "1wk"):
      async with ctx.typing():
                buf, err = create_chart(symbol.upper(), period, interval)
                if err:
                              await ctx.send(f"Error: {err}")
                              return
                          await ctx.send(file=discord.File(buf, filename=f"{symbol}_chart.png"))

if __name__ == "__main__":
      token = os.getenv('DISCORD_TOKEN')
    if token:
              bot.run(token)
else:
        print("No token")
