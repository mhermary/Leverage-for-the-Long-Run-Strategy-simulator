import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
import datetime
import matplotlib.dates as mdates

moving_avg = 100

ticker1 = 'CASH.TO' # ^GSPC = S&P500, ^NDX = NASDAQ, ^DJI = Dow Jones
ticker2 = 'CASH.TO' 
# Does not work well with ETFs that regularly pay dividends like BIL, SHV, VGSH, CASH.TO
# Something in the yfinance data import does not lower the stock price after dividends are paid out
period = '5y'

indx = yf.Ticker(ticker1) # ^GSPC = S&P500, ^NDX = NASDAQ, ^DJI = Dow Jones
alt = yf.Ticker(ticker2) # ^GSPC = S&P500, ^NDX = NASDAQ, ^DJI = Dow Jones
data = indx.history(period = period) # Index data to track, '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
alt_data = alt.history(period = period) # Index data to track
# Hedge to be rotated into when indx is under moving average. Only rotates in if cash_only is false - < 3mo tbill BIL, SHV, VGSH; Gold GLD, IAU; market short: SPDN, PSQ
# start = datetime.datetime(2019, 7, 1)
# end = datetime.datetime.now()
# tbill_data = web.DataReader('TB3MS', 'fred', start, end)  # 3-month T-Bill data
# tbill_prices = tbill_data
data['MA'] = data['Close'].rolling(window=moving_avg).mean()

fig, ax1 = plt.subplots(figsize=(10, 5))
# fig, ax2 = plt.subplots(figsize=(10, 5))
plot_MA = str(moving_avg) + " day MA"
ax1.plot(data.index, data['Close'], label=ticker1, color='blue')
ax1.plot(data.index, data['MA'], label=plot_MA, color='orange')
# ax2.plot(alt_data.index, alt_data['Close'], label= 'Alt Close Price', color='red')
ax1.set_ylabel(ticker1)
ax1.set_title('Price vs Daily Moving Average')
ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as Year-Month
ax1.xaxis.set_major_locator(mdates.YearLocator())  # Major ticks at each year
ax1.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks at each month
plt.xticks(rotation=-45)  # Rotate date labels for better readability
ax2 = ax1.twinx()
# ax2.plot(data.index, capital_history, label='Capital', color='purple', linestyle='--', alpha=0.7)
ax2.plot(alt_data.index, alt_data['Close'], label= ticker2, color='red')
# ax2.plot(tbill_data.index, tbill_prices, label= ticker2, color='red')
ax2.set_ylabel(ticker2)
ax2.legend(loc='upper right')
plt.show()