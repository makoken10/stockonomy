import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as dt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import talib
import requests
import tweepy
import config 
import ta
import yahoo_fin
import io
import requests
import requests_html
import yahoo_fin.stock_info as si
from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.pylab import date2num
import altair as alt
import cufflinks as cf

auth = tweepy.OAuthHandler(config.TWITTER_CONSUMER_KEY, config.TWITTER_CONSUMER_SECRET)
auth.set_access_token(config.TWITTER_ACCESS_TOKEN, config.TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

#sidebar menu
option = st.sidebar.selectbox("Which Dashboard?", ('Tests','Overview','Chart2','Twitter','Income Statement','Fundamentals', 'StockTwits', 'Chart', 'Pattern','Technicals','Analyst Recommendations','Balance Sheet'), 1)

#sidebar output = $XXXX
ticker = st.sidebar.text_input("Symbol", value='SQ', max_chars=5)

#yfinance inputs
stock = yf.Ticker(ticker)
info = stock.info

if option == 'Fundamentals':
    st.subheader(info['longName'])
    st.markdown('** Sector **: ' + info['logo_url']) 
    st.markdown('** Sector **: ' + info['sector'])
    st.markdown('** Industry **: ' + info['industry'])
    st.markdown('** Country **: ' + info['country'])
    st.markdown('** Website **: ' + info['website'])
    st.markdown('** Business Summary **')
    st.info(info['longBusinessSummary'])

    quote_table = si.get_quote_table(ticker, dict_result=False)
    st.table(quote_table)

        
    fundInfo = {
            'Enterprise Value (USD)': info['enterpriseValue'],
            'Enterprise To Revenue Ratio': info['enterpriseToRevenue'],
            'Enterprise To Ebitda Ratio': info['enterpriseToEbitda'],
            'Net Income (USD)': info['netIncomeToCommon'],
            'Profit Margin Ratio': info['profitMargins'],
            'Forward PE Ratio': info['forwardPE'],
            'PEG Ratio': info['pegRatio'],
            'Price to Book Ratio': info['priceToBook'],
            'Forward EPS (USD)': info['forwardEps'],
            'Beta ': info['beta'],
            'Book Value (USD)': info['bookValue'],
            'Dividend Rate (%)': info['dividendRate'], 
            'Dividend Yield (%)': info['dividendYield'],
            'Five year Avg Dividend Yield (%)': info['fiveYearAvgDividendYield'],
            'Payout Ratio': info['payoutRatio']
        }
    
    fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
    fundDF = fundDF.rename(columns={0: 'Value'})
    st.subheader('Fundamental Info') 
    st.table(fundDF)
    
    marketInfo = {
            "Volume": info['volume'],
            "Average Volume": info['averageVolume'],
            "Market Cap ($B)": info["marketCap"]/1000000000,
            "Float Shares": info['floatShares'],
            "Regular Market Price (USD)": info['regularMarketPrice'],
            'Bid Size': info['bidSize'],
            'Ask Size': info['askSize'],
            "Share Short": info['sharesShort'],
            'Short Ratio': info['shortRatio'],
            'Share Outstanding': info['sharesOutstanding']
    
        }
    marketDF = pd.DataFrame.from_dict(marketInfo, orient='index')
    marketDF = marketDF.rename(columns={0: 'Value'})
    st.subheader('Market Info') 
    st.table(marketDF)

if option == 'Technicals':

    def calcMovingAverage(data, size):
        df = data.copy()
        df['sma'] = df['Adj Close'].rolling(size).mean()
        df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
        df.dropna(inplace=True)
        return df
    
    def calc_macd(data):
        df = data.copy()
        df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
        df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
        df.dropna(inplace=True)
        return df

    def calcBollinger(data, size):
        df = data.copy()
        df["sma"] = df['Adj Close'].rolling(size).mean()
        df["bolu"] = df["sma"] + 2*df['Adj Close'].rolling(size).std(ddof=0) 
        df["bold"] = df["sma"] - 2*df['Adj Close'].rolling(size).std(ddof=0) 
        df["width"] = df["bolu"] - df["bold"]
        df.dropna(inplace=True)
        return df

    st.title('Technical Indicators')
    st.subheader('Moving Average')
    
    coMA1, coMA2 = st.beta_columns(2)
    
    with coMA1:
        numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)    
    
    with coMA2:
        windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)  
        
    start = dt.datetime.today()-dt.timedelta(numYearMA * 365)
    end = dt.datetime.today()
    dataMA = yf.download(ticker,start,end)
    df_ma = calcMovingAverage(dataMA, windowSizeMA)
    df_ma = df_ma.reset_index()

    figMA = go.Figure()
    
    figMA.add_trace(
            go.Scatter(
                    x = df_ma['Date'],
                    y = df_ma['Adj Close'],
                    name = "Prices Over Last " + str(numYearMA) + " Year(s)"
                )
        )
    
    figMA.add_trace(
                go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['sma'],
                        name = "SMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
                    )
            )
    
    figMA.add_trace(
                go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['ema'],
                        name = "EMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
                    )
            )
    
    figMA.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    
    figMA.update_layout(legend_title_text='Trend')
    figMA.update_yaxes(tickprefix="$")
    
    st.plotly_chart(figMA, use_container_width=True)  
    
    st.subheader('Moving Average Convergence Divergence (MACD)')
    numYearMACD = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=2) 
    
    startMACD = dt.datetime.today()-dt.timedelta(numYearMACD * 365)
    endMACD = dt.datetime.today()
    dataMACD = yf.download(ticker,startMACD,endMACD)
    df_macd = calc_macd(dataMACD)
    df_macd = df_macd.reset_index()
    
    figMACD = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.01)
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['Adj Close'],
                    name = "Prices Over Last " + str(numYearMACD) + " Year(s)"
                ),
            row=1, col=1
        )
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['ema12'],
                    name = "EMA 12 Over Last " + str(numYearMACD) + " Year(s)"
                ),
            row=1, col=1
        )
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['ema26'],
                    name = "EMA 26 Over Last " + str(numYearMACD) + " Year(s)"
                ),
            row=1, col=1
        )
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['macd'],
                    name = "MACD Line"
                ),
            row=2, col=1
        )
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['signal'],
                    name = "Signal Line"
                ),
            row=2, col=1
        )
    
    figMACD.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0
    ))
    
    figMACD.update_yaxes(tickprefix="$")
    st.plotly_chart(figMACD, use_container_width=True)
    
    st.subheader('Bollinger Band')
    coBoll1, coBoll2 = st.beta_columns(2)
    with coBoll1:
        numYearBoll = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=6) 
        
    with coBoll2:
        windowSizeBoll = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=7)
    
    startBoll= dt.datetime.today()-dt.timedelta(numYearBoll * 365)
    endBoll = dt.datetime.today()
    dataBoll = yf.download(ticker,startBoll,endBoll)
    df_boll = calcBollinger(dataBoll, windowSizeBoll)
    df_boll = df_boll.reset_index()
    figBoll = go.Figure()
    figBoll.add_trace(
            go.Scatter(
                    x = df_boll['Date'],
                    y = df_boll['bolu'],
                    name = "Upper Band"
                )
        )
    
    
    figBoll.add_trace(
                go.Scatter(
                        x = df_boll['Date'],
                        y = df_boll['sma'],
                        name = "SMA" + str(windowSizeBoll) + " Over Last " + str(numYearBoll) + " Year(s)"
                    )
            )
    
    
    figBoll.add_trace(
                go.Scatter(
                        x = df_boll['Date'],
                        y = df_boll['bold'],
                        name = "Lower Band"
                    )
            )
    
    figBoll.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0
    ))
    
    figBoll.update_yaxes(tickprefix="$")
    st.plotly_chart(figBoll, use_container_width=True)  

if option == 'Chart':
    # Set sample stock symbol to instrument variable

    API_URL = "https://www.alphavantage.co/query"

    data = { "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize" : "compact",
        "datatype": "json",
        "apikey": "STTFPTOT86P19ZDY" } #ENTER YOUR ALPHAVANTAGE KEY HERE

    #https://www.alphavantage.co/query/
    response = requests.get(API_URL, data).json()

    data = pd.DataFrame.from_dict(response['Time Series (Daily)'], orient= 'index').sort_index(axis=1)
    data = data.rename(columns={ '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data['Date'] = data.index

    fig = go.Figure(data=[go.Ohlc(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=ticker)])
    fig.update(layout_xaxis_rangeslider_visible=False)

    fig.update_layout(
        title=ticker+ ' Daily Chart',
        xaxis_title="Date",
        yaxis_title="Price ($)",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="black"
        )
    )

    st.plotly_chart(fig,  use_container_width=True)

if option == 'StockTwits':

    r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json")

    data = r.json()

    for message in data['messages']:
        st.image(message['user']['avatar_url'])
        st.write(message['user']['username'])
        st.write(message['created_at'])
        st.write(message['body'])

if option == 'Income Statement':
    income_statement = si.get_income_statement(ticker)
    st.table(income_statement)
    
if option == 'Overview':
    
    # get basic stock info
    quote = si.get_quote_table(ticker)  
    st.header(ticker)
    st.subheader(info['longName'])
    st.subheader(info['regularMarketPrice']) 
    st.markdown(info['marketCap']/1000000000)  
    st.markdown('** Sector **: ' + info['sector']) 
    st.markdown('** Industry **: ' + info['industry'])
    st.markdown('** Country **: ' + info['country'])
    st.markdown('** Website **: ' + info['website'])

    # get revenue bar chart
    st.header('Revenues')
    income_statement = si.get_income_statement(ticker,yearly = True)
    revenue = income_statement.loc['totalRevenue']
    revenue.index = pd.to_datetime(revenue.index, format = '%Y-%m-%d').strftime('%Y')
    st.bar_chart(revenue)    

    # get cash bar chart
    st.header('Cash')
    balance_sheet = si.get_balance_sheet(ticker,yearly = True)
    cash = balance_sheet.loc['cash']
    cash.index = pd.to_datetime(revenue.index, format = '%Y-%m-%d').strftime('%Y')
    st.bar_chart(cash)    
    
    #return on equity

    #research & development
    st.header('Research & Development')
    research = income_statement.loc['researchDevelopment']
    research.index = pd.to_datetime(research.index, format = '%Y-%m-%d').strftime('%Y')
    st.bar_chart(research)    


    #earnings?
    st.header('Earnings?')
    earn = balance_sheet.loc['retainedEarnings']
    earn.index = pd.to_datetime(earn.index, format = '%Y-%m-%d').strftime('%Y')
    st.bar_chart(earn)  

    # get earnings history for AAPL
    st.header('Earnings')
    aapl_earnings_hist = si.get_earnings_history("aapl")
    frame <- pd.DataFrame.from_dict(aapl_earnings_hist)

if option == 'Analyst Recommendations':
    stock = yf.Ticker(ticker)
    st.subheader("""**Analysts recommendation** for """ + ticker)
    display_analyst_rec = (stock.recommendations)
    if display_analyst_rec.empty == True:
        st.write("No data available at the moment")
    else:
        st.write(display_analyst_rec)

if option == 'Balance Sheet':
    balance_sheet = si.get_balance_sheet(ticker)
    st.table(balance_sheet)

if option == 'Twitter':
    
    for username in config.TWITTER_USERNAMES:
        user = api.get_user(username)
        tweets = api.user_timeline(username)

        st.subheader(username)
        st.image(user.profile_image_url)
        
        for tweet in tweets:
            if '$' in tweet.text:
                words = tweet.text.split(' ')
                for word in words:
                    if word.startswith('$') and word[1:].isalpha():
                        symbol = word[1:]
                        st.write(symbol)
                        st.write(tweet.text)
                        st.image(f"https://finviz.com/chart.ashx?t={symbol}")

if option == 'Chart2':
    ticker_df = yf.download(ticker, 
                      start='2016-01-01', 
                      end='2021-03-31', 
                      progress=False)
    ticker_df

    def get_indicators(data):
        # Get MACD
        data["macd"], data["macd_signal"], data["macd_hist"] = talib.MACD(data['Close'])
    
        # Get MA10 and MA30
        data["ma10"] = talib.MA(data["Close"], timeperiod=10)
        data["ma30"] = talib.MA(data["Close"], timeperiod=30)
    
        # Get RSI
        data["rsi"] = talib.RSI(data["Close"])
        return data
    ticker_df2 = get_indicators(ticker_df)
    ticker_df2

    def plot_chart(data, n, ticker):
    
        # Filter number of observations to plot
        data = data.iloc[-n:]
    
        # Create figure and set axes for subplots
        fig = plt.figure()
        fig.set_size_inches((20, 16))
        ax_candle = fig.add_axes((0, 0.72, 1, 0.32))
        ax_macd = fig.add_axes((0, 0.48, 1, 0.2), sharex=ax_candle)
        ax_rsi = fig.add_axes((0, 0.24, 1, 0.2), sharex=ax_candle)
        ax_vol = fig.add_axes((0, 0, 1, 0.2), sharex=ax_candle)
    
        # Format x-axis ticks as dates
        ax_candle.xaxis_date()
    
        # Get nested list of date, open, high, low and close prices
        ohlc = []
        for date, row in data.iterrows():
            openp, highp, lowp, closep = row[:4]
            ohlc.append([date2num(date), openp, highp, lowp, closep])
 
        # Plot candlestick chart
        ax_candle.plot(data.index, data["ma10"], label="MA10")
        ax_candle.plot(data.index, data["ma30"], label="MA30")
        candlestick_ohlc(ax_candle, ohlc, colorup="g", colordown="r", width=0.8)
        ax_candle.legend()
    
        # Plot MACD
        ax_macd.plot(data.index, data["macd"], label="macd")
        ax_macd.bar(data.index, data["macd_hist"] * 3, label="hist")
        ax_macd.plot(data.index, data["macd_signal"], label="signal")
        ax_macd.legend()
    
        # Plot RSI
        # Above 70% = overbought, below 30% = oversold
        ax_rsi.set_ylabel("(%)")
        ax_rsi.plot(data.index, [70] * len(data.index), label="overbought")
        ax_rsi.plot(data.index, [30] * len(data.index), label="oversold")
        ax_rsi.plot(data.index, data["rsi"], label="rsi")
        ax_rsi.legend()
    
        # Show volume in millions
        ax_vol.bar(data.index, data["Volume"] / 1000000)
        ax_vol.set_ylabel("(Million)")
   
        # Save the chart as PNG
        fig.savefig("charts/" + ticker + ".png", bbox_inches="tight")
    
        st.pyplt(fig)

        plot_chart(ticker_df2, 180, ticker)

if option == 'Tests':
    
    #http://theautomatic.net/yahoo_fin-documentation/#get_earnings
    valuation = si.get_stats_valuation(ticker)
    valuation

    financials = si.get_financials(ticker)
    st.write(financials)

    earn = si.get_earnings(ticker)
    earn
    st.table(earn)

    prices = si.get_data(ticker)
    prices

    analysts = si.get_analysts_info(ticker)
    analysts
    st.table(analysts)




