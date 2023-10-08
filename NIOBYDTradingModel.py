import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import yfinance as yf

df= yf.download(["NIO","BYDDF"], start="2020-01-01", end="2023-01-01")
plt.figure(figsize=(15,7))
plt.plot(df["Adj Close"])
plt.legend(["NIO","BYDDF"])
plt.ylabel("Price[$]")
plt.xlabel("Date")
plt.grid()
plt.ylabel("Data")

# check for nan vlaues, duplicated values, outliers

print("nan vlaues",df["Adj Close"].isnull().sum())
print("duplicates",df["Adj Close"].duplicated().sum())
print("data type",df["Adj Close"].dtypes)



### calculate daily returns
BYD_DR=((df["Adj Close"]["BYDDF"]).pct_change()).dropna()
NIO_DR=(df["Adj Close"]["NIO"].pct_change()).dropna()

plt.figure(figsize=(15,7))
plt.scatter(BYD_DR.index, BYD_DR, alpha=0.5, color="black")
plt.scatter(NIO_DR.index, NIO_DR, alpha=0.5, color="red")
plt.legend(["BYD","NIO"])
plt.ylabel("Daily Returns")
plt.xlabel("Date")
plt.grid()
plt.show()


# calculate 20, 40 and 60 days moving averages
mv_20=(df["Adj Close"].rolling(window=20).mean())
mv_40=(df["Adj Close"].rolling(window=40).mean())
mv_60=(df["Adj Close"].rolling(window=60).mean())


## plot NIO stock plus 20, 40 and 60 day moving averages

plt.figure(figsize=(15,7))
plt.plot(df["Adj Close"]["NIO"])
plt.plot(mv_20["NIO"])
plt.plot(mv_40["NIO"])
plt.plot(mv_60["NIO"])
plt.xlabel("Date")
plt.ylabel("Price[$]")
plt.legend(["NIO Stock","20 days MA", "40 days MA", "60 days MA"])
plt.grid()


plt.figure(figsize=(15,7))
plt.plot(df["Adj Close"]["BYDDF"])
plt.plot(mv_20["BYDDF"])
plt.plot(mv_40["BYDDF"])
plt.plot(mv_60["BYDDF"])
plt.xlabel("Date")
plt.ylabel("Price[$]")
plt.legend(["BYD Stock","20 days MV","40 days MA","60 days MA"])
plt.grid()


#calculate performance metrics sharpe ratio, annual returns, total returns


### sharpe ratio

## sharpe ratio

avgDailyReturnsNIO=NIO_DR.mean()
annualreturnsNIO=(1+avgDailyReturnsNIO)**252 -1
avgDailyVolatilityNIO=NIO_DR.std()
annualVolatilityNIO=avgDailyVolatilityNIO*np.sqrt(252)
sharpRatioNio= (annualreturnsNIO-0.04)/annualVolatilityNIO

avgDailyReturnsBYD=BYD_DR.mean()
annualreturnsBYD=(1+avgDailyReturnsBYD)**252 -1
avgDailyVolatilityBYD=BYD_DR.std()
annualVolatilityBYD=avgDailyVolatilityBYD*np.sqrt(252)
sharpeRatioBYD=(annualreturnsBYD-0.04)/annualVolatilityBYD
print("NIO Sharpe Ratio",sharpRatioNio)
print("BYD Sharpe Ratio",sharpeRatioBYD)


## annualized Returns

print("annual returns NIO",annualreturnsNIO)
print("annual returns BYD",annualreturnsBYD)


## calculate Beta:

sp500 = yf.download("^GSPC", start="2020-01-01", end="2023-01-01")["Adj Close"]

sp500DR=sp500.pct_change().dropna()
NIO_SP500=pd.concat([NIO_DR,sp500DR],axis=1).dropna()
NIO_SP500.columns=["NIO","SP500"]
BYD_SP500=pd.concat([BYD_DR,sp500DR],axis=1).dropna()
BYD_SP500.columns=["BYD","SP500"]

### Beta NIO/SP500 and BYD/SP500 
NIO_SP500_Beta=(NIO_SP500.cov().iloc[0,1])/(NIO_SP500["SP500"].var())
BYD_SP500_Beta=(BYD_SP500.cov().iloc[0,1])/(BYD_SP500["SP500"].var())
print("NIO_SP500 Beta",NIO_SP500_Beta)
print("BYD_SP500 Beta",BYD_SP500_Beta)

## corrleation NIO/BYD
corrNIOBYD=df["Adj Close"].corr().iloc[0,1]
print("corrleation NIO/BYD",corrNIOBYD)


## drawdown

cumMaxNIO=df["Adj Close"]["NIO"].cummax()
dailydrawdownNIO=(df["Adj Close"]["NIO"]-cumMaxNIO)/cumMaxNIO
maxdrawdownNIO=dailydrawdownNIO.min()
cumMaxBYD=df["Adj Close"]["BYDDF"].cummax()
dailydrawdownBYD=(df["Adj Close"]["BYDDF"]-cumMaxBYD)/cumMaxBYD
maxdrawdownBYD=dailydrawdownBYD.min()
print("NIO drawdown",maxdrawdownNIO)
print("BYD drawdown",maxdrawdownBYD)


## Value at Risk
returns = df["Adj Close"].pct_change().dropna()
var_95 = returns.quantile(0.05)
print("VaR (0.95)",var_95)




### generate Trading singal 

### NIO trading signals
signals = pd.DataFrame(index=df.index)
signals['Price'] = df['Adj Close']['NIO']
signals["Short Term MA"]=mv_20["NIO"]
signals["Long Term MA"]=mv_60["NIO"]
signals=signals.dropna()


signals['Above'] = np.where(signals['Short Term MA'] > signals['Long Term MA'], 1.0, 0.0)
signals['Signal'] = signals['Above'].diff()




## plots


plt.figure(figsize=(15,7))

plt.plot(signals['Price'], label='NIO Price', alpha=0.6)
plt.plot(signals['Short Term MA'], label='20-day MA', alpha=0.6)
plt.plot(signals['Long Term MA'], label='60-day MA', alpha=0.6)
plt.plot(signals[signals['Signal'] == 1.0].index, 
         signals['Short Term MA'][signals['Signal'] == 1.0], 
         '^', markersize=10, color='g', label='BUY Signal', lw=0)
plt.plot(signals[signals['Signal'] == -1.0].index, 
         signals['Short Term MA'][signals['Signal'] == -1.0], 
         'v', markersize=10, color='r', label='SELL Signal', lw=0)
plt.legend(loc='best')
plt.grid(True)
plt.title('NIO Price and Moving Averages with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
signalBYD=pd.DataFrame(index=df.index)
signalBYD["Price"]=df["Adj Close"]["BYDDF"]
signalBYD["Short_MA"]=mv_20["BYDDF"]
signalBYD["Long_MA"]=mv_60["BYDDF"]
signalBYD=signalBYD.dropna()




signalBYD["Above"]=np.where(signalBYD["Short_MA"]>signalBYD["Long_MA"],1.0,0.0)
signalBYD["Signals"]=signalBYD["Above"].diff()
plt.figure(figsize=(15,7))
plt.plot(signalBYD['Price'], label='BYD Price', alpha=0.6)
plt.plot(signalBYD['Short_MA'], label='20-day MA', alpha=0.6)
plt.plot(signalBYD['Long_MA'], label='60-day MA', alpha=0.6)
plt.plot(signalBYD[signalBYD['Signals'] == 1.0].index, 
         signalBYD['Short_MA'][signalBYD['Signals'] == 1.0], 
         '^', markersize=10, color='g', label='BUY Signal', lw=0)
plt.plot(signalBYD[signalBYD['Signals'] == -1.0].index, 
         signalBYD['Short_MA'][signalBYD['Signals'] == -1.0], 
         'v', markersize=10, color='r', label='SELL Signal', lw=0)  # Added 'v', color='r', label='SELL Signal'
plt.legend(loc='best')
plt.grid(True)
plt.title('BYD Price and Moving Averages with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
