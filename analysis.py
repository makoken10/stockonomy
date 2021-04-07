import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as dt
from datetime import date
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import tweepy
import config 
import yahoo_fin
import io
import requests
import requests_html
import yahoo_fin.stock_info as si
from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.pylab import date2num
import altair as alt
import cufflinks as cf
import finplot as fplt
import streamlit.components.v1 as components
import os
import sys
from bokeh.plotting import figure


# check if the library folder already exists, to avoid building everytime you load the pahe
if not os.path.isdir("/tmp/ta-lib"):

    # Download ta-lib to disk
    with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
        response = requests.get(
            "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        )
        file.write(response.content)
    # get our current dir, to configure it back again. Just house keeping
    default_cwd = os.getcwd()
    os.chdir("/tmp")
    # untar
    os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
    os.chdir("/tmp/ta-lib")
    os.system("ls -la /app/equity/")
    # build
    os.system("./configure --prefix=/home/appuser")
    os.system("make")
    # install
    os.system("make install")
    # bokeh sample data
    os.system("bokeh sampledata")
    # install python package
    os.system(
        'pip3 install --global-option=build_ext --global-option="-L/home/appuser/lib/" --global-option="-I/home/appuser/include/" ta-lib'
    )
    # back to the cwd
    os.chdir(default_cwd)
    print(os.getcwd())
    sys.stdout.flush()

# add the library to our current environment
from ctypes import *

lib = CDLL("/home/appuser/lib/libta_lib.so.0")
# import library
import talib


auth = tweepy.OAuthHandler(config.TWITTER_CONSUMER_KEY, config.TWITTER_CONSUMER_SECRET)
auth.set_access_token(config.TWITTER_ACCESS_TOKEN, config.TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

#sidebar menu
option = st.sidebar.selectbox("Which Dashboard?", ('Overview','Fundamentals','Technicals','Chart','Twitter','StockTwits','Balance Sheet','Income Statement','Cash Flow'), 2)

#sidebar output = $XXXX
ticker = st.sidebar.text_input("Symbol", value='SQ', max_chars=5)

#yfinance inputs
stock = yf.Ticker(ticker)
info = stock.info

today = date.today()

if option == 'Fundamentals':
    st.subheader(info['longName'])

    # get revenue bar chart
    st.header('Revenues')

    st.markdown('Price / Sales:')  
    
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
    #PEG <1
    st.header('Earnings?')

    st.markdown('Trailing P/E:')

    st.markdown('Forward P/E:')

    st.markdown('Price / Earnings-to-Growth:')

    # get earnings history for AAPL
    earnings = si.get_earnings_history(ticker)
    st.table(earnings)

    quote_table = si.get_quote_table(ticker, dict_result=False)
    st.table(quote_table)

    
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
    
    st.header(ticker)
    
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

    #overview
    st.header(ticker)
    st.subheader(info['longName'])
    st.subheader(info['regularMarketPrice'])
    st.markdown('Analyst Target:')
    st.markdown(quote['1y Target Est'])
    st.markdown(info['marketCap']/1000000000) 
    st.markdown('** Sector **: ' + info['sector']) 
    st.markdown('** Industry **: ' + info['industry'])
    st.markdown('** Country **: ' + info['country'])
    st.markdown('** Website **: ' + info['website'])
    st.markdown('** Business Summary **')
    st.info(info['longBusinessSummary'])

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

if option == 'Cash Flow':
    flow = si.get_cash_flow(ticker)
    st.table(flow)

if option == 'Buy Zone':
    df = stock.history(start="2020-01-01", end='2020-09-04')
    st.table(df.head())
    
    arr = np.random.normal(1, 1, size=100)

    plt.hist(arr, bins=20)

if option =='Chart':
    components.html("""
        <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
    <div id="tradingview_972e6"></div>
    <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/symbols/NASDAQ-AAPL/" rel="noopener" target="_blank"><span class="blue-text">AAPL Chart</span></a> by TradingView</div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
    new TradingView.widget(
    {
    "width": '100%',
    "height": 610,
    "symbol": "NASDAQ:AAPL",
    "interval": "D",
    "timezone": "Etc/UTC",
    "theme": "light",
    "style": "0",
    "locale": "en",
    "toolbar_bg": "#f1f3f6",
    "hide_side_toolbar": false,
    "enable_publishing": false,
    "allow_symbol_change": true,
    "details": false,
    "studies": [
        "MAExp@tv-basicstudies",
        "RSI@tv-basicstudies",
        "StochasticRSI@tv-basicstudies",
    ],
    "container_id": "tradingview_972e6"
    }
    );
    </script>
    </div>
        <!-- TradingView Widget END -->
                """,
    height=1000000)

