import pandas as pd
from stockstats import StockDataFrame as Sdf

# df = pd.read_csv('/home/shainshahid/Rl/StockPredictor/data/GOOGLE.csv')
# df['ticker'] = 'google' 

# df1 = pd.read_csv('/home/shainshahid/Rl/StockPredictor/data/APPLE.csv')
# df1['ticker'] = 'apple'

# print(df1.shape)
# print(df1.head())
# def remove(date):
#     date = date.replace('-','')
#     return date

# new_df = pd.concat([df,df1], ignore_index=True)
# new_df['Date'] = new_df['Date'].apply(lambda x: remove(x))
# print(new_df.head())
# new_df.to_csv('/home/shainshahid/Rl/StockPredictor/data/apple_google.csv')

df = pd.read_csv('/home/shainshahid/Rl/StockPredictor/data/apple_google.csv')

def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    # data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    # data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    # data['adjcp'] = data['prccd'] / data['ajexdi']
    # data['open'] = data['prcod'] / data['ajexdi']
    # data['high'] = data['prchd'] / data['ajexdi']
    # data['low'] = data['prcld'] / data['ajexdi']
    # data['volume'] = data['cshtrd']


    # data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    # data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    #data = data[['Date','Open','High','Low','Close','AdjClose','Volume','ticker']]
    data = data.sort_values(['ticker', 'Date'], ignore_index=True)
    return data

def process_data(df):

    df = df.sort_values(['ticker', 'Date'], ignore_index=True)

    stock = Sdf.retype(df.copy())
    print(stock.head())

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




# get data after 2009
df = df[df.Date>=20090000]
# calcualte adjusted price
df_preprocess = calcualte_price(df)
# add technical indicators using stockstats
df_final=process_data(df_preprocess)
# fill the missing values at the beginning
df_final.fillna(method='bfill',inplace=True)

print(df_final.head())
