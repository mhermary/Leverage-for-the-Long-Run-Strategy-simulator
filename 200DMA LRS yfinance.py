import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import yfinance as yf
import matplotlib.dates as mdates

moving_avg = 100
cash_only = 1

mrkt = '^GSPC'# index to follow - ^GSPC = S&P500, ^NDX = NASDAQ, ^DJI = Dow Jones
levETF = 'UPRO' # leveraged ETF to rotate into when the underlying index trades above the moving average
hedge = 'SPDN' # Hedge to be rotated into when indx is under moving average. Only rotates in if cash_only is false - < 3mo tbill BIL, SHV, VGSH; Gold GLD, IAU; market short: SPDN, PSQ
# Does not work well with ETFs that regularly pay dividends like BIL, SHV, VGSH, CASH.TO
# Something in the yfinance data import does not lower the stock price after dividends are paid out
indx = yf.Ticker(mrkt) 
letf = yf.Ticker(levETF)
tbill = yf.Ticker(hedge)
data = indx.history(period= 'max') # Index data to track
data_tr = letf.history(period= 'max')
data_tbill = tbill.history(period= 'max')
# for custom date range
# data = indx.history(start = "2005-07-01", end = "2020-12-31")
# data_tr = letf.history(start = "2007-07-01", end = "2020-12-31")
# data_tbill = tbill.history(start = "2007-07-01", end = "2020-12-31")

#calcualte MA before aligning start dates to have MA at starting data
close_prices = data[['Close']]
data['MA'] = close_prices['Close'].rolling(window=moving_avg).mean()

# setting all dataframes to have same start date
dfs = [data, data_tr, data_tbill]
start_dates = [df.index[0] for df in dfs]
latest_start_date = max(start_dates)
finish_date = data.index[-1]
aligned_dfs = [df[df.index >= latest_start_date] for df in dfs]
data = aligned_dfs[0]
data_tr = aligned_dfs[1]
data_tbill = aligned_dfs[2]

# Buy/Sell when price is above/below the moving average
data_tr['Buy'] = (data['Close'] > data['MA'])
data_tr['Sell'] = (data['Close'] < data['MA'])

# Backtest
position = 0  # ETF Shares
tb_position = 0 # tbill Shares
capital = 10000  # initial capital
capital_history = []
returns = []
buydates = []
selldates = []
trades = 0
holding_tbill = False
in_tbill_date =  0 if cash_only == 1 else 1

if cash_only:
    for index, row in data_tr.iterrows():
        # Buy
        if row['Buy'] and position == 0:
            buy_price = row['Close']
            position = capital / buy_price  # Buy the asset
            capital = 0
            buydates.append(index)
            trades = trades + 1

        # Sell
        elif row['Sell'] and position > 0:
            sell_price = row['Close']
            capital = position * sell_price  # Sell the asset
            position = 0
            # Calculate return and store it
            returns.append((sell_price - buy_price) / buy_price)
            selldates.append(index)
            trades = trades + 1

        if position > 0:
            capital_history.append(position * row['Close'])  # Track unrealized capital (still holding)
        else:
            capital_history.append(capital)  # Track realized capital (not holding)
else:
    for index, row in data_tr.iterrows():
        # Buy signal
        if row['Buy'] and position == 0:
            if holding_tbill:
                sell_price = data_tbill['Close'].loc[index]
                capital += tb_position * sell_price
                holding_tbill = False
            # capital += tb_position*data_tbill['Dividends'].loc[index] # most hedges gave minimal dividends, code is kept in case dividend hedge is chosen
            buy_price = row['Close']
            position = capital / buy_price
            capital = 0
            buydates.append(index)
            trades = trades + 1
        # Sell signal
        elif row['Sell'] and position > 0:
            sell_price = row['Close']
            capital = position * sell_price  # Sell the asset
            position = 0
            # capital += tb_position*data_tbill['Dividends'].loc[index]
            # Calculate return and store it
            if not(holding_tbill):
                buy_price = data_tbill['Close'].loc[index]
                tb_position = capital / buy_price
                capital = 0
                holding_tbill = True
            returns.append((sell_price - buy_price) / buy_price)
            selldates.append(index)
            trades = trades + 1
        if position > 0:
            capital_history.append(position * row['Close'])  # Track unrealized capital (still holding)
        elif in_tbill_date:
            capital_history.append(tb_position * data_tbill.loc[index, 'Close']) 
        else: #cash
            capital_history.append(capital)  # Track realized capital (not holding)

print(capital_history)
data['Capital'] = capital_history
# Calculate the performance
if position > 0:
    sell_price = data_tr['Close'].iloc[-1] #iloc [-1] gives last value in <CLOSE>. ILOC + Integer location based indexing
    capital = position * sell_price #Liquidate assets to see final gains
    print(f"\n Final capital : {capital:.2f}")
gains = (capital - 10000)/10000
print(f"\n Total Return: {gains * 100:.2f}%")

print(f'Start date : ', latest_start_date)
print(f'End date : ', finish_date)
num_years = (finish_date - latest_start_date).days / 365.25  # Use 365.25 for leap years
print(f"\n Number of years in the dataset: {num_years:.2f}")
if capital == 0:
    print("All capital lost - Way to go...")
    capital += 1
cagr = ((math.e**(math.log(capital/10000)/num_years))-1)*100
print(f" CAGR : {cagr :.2f}%")

print(f"\n Total number of trades : " + str(trades))
trades_per_year = trades/num_years
print(f" Trades per year : {trades_per_year:.2f}") 
print()


# print("buydates : ")
# for i in buydates:
#     print(i)

# print("selldates : ")
# for j in selldates:
#     print(j)

# in case trades want to be kept to be reviewed externally
# data['Transaction'] = data['Buy'] | data['Sell']
# df = pd.DataFrame(data)
# df.to_csv('50DMA_trades.csv', sep =',', index=False)

if 1:
    # new
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig, ax3 = plt.subplots(figsize=(10, 5))
    plot_MA = str(moving_avg) + " day MA"
    # Plot Close Price and 50-Day Moving Average on the first y-axis (ax1)
    ax1.plot(data.index, data['Close'], label= mrkt, color='blue')
    ax1.plot(data.index, data['MA'], label=plot_MA, color='orange')
    # ax1.scatter(buydates, data.loc[buydates]['Close'], marker='^', color='g', label='Buy Signal', alpha=1)
    # ax1.scatter(selldates, data.loc[selldates]['Close'], marker='v', color='r', label='Sell Signal', alpha=1)
    # Label for the first y-axis (for price)
    ax1.set_ylabel('Index Value')
    ax1.set_title('Moving Average LRS with Capital Over Time')
    ax1.legend(loc='upper left')
    # Set dates to X-Axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as Year-Month
    ax1.xaxis.set_major_locator(mdates.YearLocator())  # Major ticks at each year
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks at each month
    # plt.xticks(rotation=45)  # Rotate date labels for better readability

    # Create a second y-axis (ax2) for capital history
    ax2 = ax1.twinx()
    ax2.plot(data.index, capital_history, label='Capital', color='purple', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Capital')
    ax2.legend(loc='upper right')

    # Create a second graph (ax3) for hedge history
    ax3.plot(data.index, data_tbill['Close'], label= hedge, color='red')
    ax3.set_ylabel('Price')
    ax3.set_title('Hedge Price')
    ax3.legend(loc='upper left')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as Year-Month
    ax3.xaxis.set_major_locator(mdates.YearLocator())  # Major ticks at each year
    ax3.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks at each month
    # Label for the second y-axis (for capital)

    plt.show()