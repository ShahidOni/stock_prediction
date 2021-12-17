import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

DATASET = "apple_google.csv"

def load_dataset(*, file_name: str) -> pd.DataFrame:

    _data = pd.read_csv(file_name)
    return _data

def data_split(df,start,end):

    data = df[(df.Date >= start) & (df.Date < end)]
    data=data.sort_values(['Date','ticker'],ignore_index=True)
    data.index = data.Date.factorize()[0]
    return data


def process_data():
    df = pd.read_csv(DATASET)
    df = df.sort_values(['ticker', 'Date'], ignore_index=True)

    stock = Sdf.retype(df.copy())

    stock['close'] = stock['adjclose']
    unique_ticker = stock.ticker.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()


    for i in range(len(unique_ticker)):

        temp_macd = stock[stock.ticker == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        
        temp_rsi = stock[stock.ticker == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
  
        temp_cci = stock[stock.ticker == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
  
        temp_dx = stock[stock.ticker == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    df.fillna(method='bfill',inplace=True)

    return df

def add_turbulence(df):

    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='Date')
    df = df.sort_values(['Date','ticker']).reset_index(drop=True)
    return df



def calcualte_turbulence(df):
    
    df_price_pivot=df.pivot(index='Date', columns='ticker', values='AdjClose')
    unique_date = df.Date.unique()

    start = 252
    turbulence_index = [0]*start

    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)
    
    
    turbulence_index = pd.DataFrame({'Date':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index










