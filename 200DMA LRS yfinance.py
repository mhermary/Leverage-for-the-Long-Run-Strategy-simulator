import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import yfinance as yf
import matplotlib.dates as mdates

# When making trades, sell first then buy
# include TMF on contrary to LETF purchase/sell
moving_avg = 100
cash_only = 1

# Step 1: Load historical price data (for example, CSV file)
indx = yf.Ticker('^GSPC') # ^GSPC = S&P500, ^NDX = NASDAQ, ^DJI = Dow Jones
letf = yf.Ticker('UPRO')
tbill = yf.Ticker('SPDN')
data = indx.history(period= 'max') # Index data to track
data_tr = letf.history(period= 'max')
data_tbill = tbill.history(period= 'max')
# data = indx.history(start = "2005-07-01", end = "2020-12-31") # Index data to track
# data_tr = letf.history(start = "2007-07-01", end = "2020-12-31")
# data_tbill = tbill.history(start = "2007-07-01", end = "2020-12-31")
# print(data)
# print(data.head())  # For S&P 500 data
# print(data_tr.head())  # For UPRO data
# print(data_tbill.head())  # For TMF data

#calcualte MA before aligning start dates to have MA at starting data
close_prices = data[['Close']]
# data['200MA'] = close_prices['Close'].rolling(window=200).mean()
# data['50MA'] = close_prices['Close'].rolling(window=50).mean()
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

# Step 2: Extract the moving average
# close_prices = data[['Close']]
# data['MA'] = close_prices['Close'].rolling(window=moving_avg).mean()

# Step 3: Create buy/sell signals
# Buy when price crosses above the moving average
# data_tr['Buy'] = (data['Close'] > data['MA']) & (data['Close'].shift(1) <= data['MA'].shift(1))
# data_tr['Buy'] = (data['Close'] > data['200MA'])
data_tr['Buy'] = (data['Close'] > data['MA'])

# Sell when price crosses below the moving average
# data_tr['Sell'] = (data['Close'] < data['MA']) & (data['Close'].shift(1) >= data['MA'].shift(1))
# data_tr['Sell'] = (data['Close'] < data['50MA'])
data_tr['Sell'] = (data['Close'] < data['MA'])

# Step 4: Backtest the strategy
# Initialize variables
position = 0  # ETF Shares
tb_position = 0 # tbill Shares
capital = 10000  # initial capital
capital_history = []
returns = []
buydates = []
selldates = []
trades = 0
# cash_only = 1
holding_tbill = False
in_tbill_date =  0 if cash_only == 1 else 1

if cash_only:
    for index, row in data_tr.iterrows():
        # Buy signal
        if row['Buy'] and position == 0:
            buy_price = row['Close']
            # tmp = data_tr.loc[index] 
            # buy_price = tmp['<CLOSE>']
            # print(f"Buying at {buy_price} on date {index}")
            # buy_price = float(buy_price)
            position = capital / buy_price  # Buy the asset
            # position = float(position)
            capital = 0
            buydates.append(index)
            trades = trades + 1

        # Sell signal
        elif row['Sell'] and position > 0:
            sell_price = row['Close']
            # tmp = data_tr.loc[index] 
            # buy_price = tmp['<CLOSE>']
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
            # capital += tb_position*data_tbill['Dividends'].loc[index]
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
# Step 5: Calculate the performance
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

# data['Transaction'] = data['Buy'] | data['Sell']
# df = pd.DataFrame(data)
# df.to_csv('50DMA_trades.csv', sep =',', index=False)

if 1:
    # Create a new figure
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig, ax3 = plt.subplots(figsize=(10, 5))
    plot_MA = str(moving_avg) + " day MA"
    # Plot Close Price and 50-Day Moving Average on the first y-axis (ax1)
    ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax1.plot(data.index, data['MA'], label=plot_MA, color='orange')
    # ax1.plot(data.index, data['200MA'], label='200-Day MA', color='orange')
    # ax1.plot(data.index, data['50MA'], label='50-Day MA', color='red')
    # ax1.scatter(buydates, data.loc[buydates]['Close'], marker='^', color='g', label='Buy Signal', alpha=1)
    # ax1.scatter(selldates, data.loc[selldates]['Close'], marker='v', color='r', label='Sell Signal', alpha=1)
    # Label for the first y-axis (for price)
    ax1.set_ylabel('Price')
    ax1.set_title('Moving Average LRS with Capital Over Time')
    ax1.legend(loc='upper left')
    #Set dates to X-Axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as Year-Month
    ax1.xaxis.set_major_locator(mdates.YearLocator())  # Major ticks at each year
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks at each month
    # plt.xticks(rotation=45)  # Rotate date labels for better readability
    # Create a second y-axis (ax2) for capital history
    ax2 = ax1.twinx()
    ax2.plot(data.index, capital_history, label='Capital', color='purple', linestyle='--', alpha=0.7)
    ax3.plot(data.index, data_tbill['Close'], label='T-Bill Price', color='red')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as Year-Month
    ax3.xaxis.set_major_locator(mdates.YearLocator())  # Major ticks at each year
    ax3.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks at each month
    # Label for the second y-axis (for capital)
    ax2.set_ylabel('Capital')
    ax2.legend(loc='upper right')
    plt.show()