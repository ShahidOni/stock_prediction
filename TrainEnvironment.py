import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from environment_config import *


class StockTrain(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df,day = 0):
        self.day = day
        self.df = df
        self.action_space = spaces.Box(low = -1, high = 1,shape = (number_of_total_stocks,)) 
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (13,))
        self.data = self.df.loc[self.day,:]
        self.terminal = False             
        self.state = [start_balance] + \
                      self.data.AdjClose.values.tolist() + \
                      [0]*number_of_total_stocks + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()

        self.reward = 0
        self.cost = 0
        self.asset_memory = [start_balance]
        self.rewards_memory = []
        self.trades = 0
        self._seed()


    def sell(self, index, action):
        # print(index)
        # print('stock price ', self.state[index])
        # print('macd',self.state[index+number_of_total_stocks+1])
        # print('rsi',self.state[index+2*number_of_total_stocks]+1)
        # print('cci',self.state[index+3*number_of_total_stocks]+1)
        # print('adx',self.state[index+4*number_of_total_stocks]+1)
        #print(self.state)
        
        macd = self.state[index+number_of_total_stocks+1]
        rsi = self.state[index+2*number_of_total_stocks]+1
        cci = self.state[index+3*number_of_total_stocks]+1
        adx = self.state[index+4*number_of_total_stocks]+1
        #print(self.state)
        if self.state[index+number_of_total_stocks+1] > 0:
            if (cci < -100 or rsi) > 70 and adx > 50:
                self.state[0] += \
                self.state[index+1]*min(abs(action),self.state[index+number_of_total_stocks+1]) 
                self.state[index+number_of_total_stocks+1] -= min(abs(action), self.state[index+number_of_total_stocks+1])
                self.cost +=self.state[index+1]*min(abs(action),self.state[index+number_of_total_stocks+1]) 
                self.trades+=1
        else:
            pass

    
    def buy(self, index, action):

        macd = self.state[index+number_of_total_stocks+1]
        rsi = self.state[index+2*number_of_total_stocks]+1
        cci = self.state[index+3*number_of_total_stocks]+1
        adx = self.state[index+4*number_of_total_stocks]+1

        if (cci > 100 or rsi < 30) and adx > 50:
            available_amount = self.state[0] // self.state[index+1]
            self.state[0] -= self.state[index+1]*min(available_amount, action)
            self.state[index+number_of_total_stocks+1] += min(available_amount, action)
            self.cost+=self.state[index+1]*min(available_amount, action)
            self.trades+=1
        
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            # print('Terminal State')
            plt.plot(self.asset_memory,'r')

            #print(len(self.asset_memory))
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(number_of_total_stocks+1)])*np.array(self.state[(number_of_total_stocks+1):(number_of_total_stocks*2+1)]))
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/train.csv')

            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                  df_total_value['daily_return'].std()

            return self.state, self.reward, self.terminal,{}

        else:

            actions = actions * total_share
            
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(number_of_total_stocks+1)])*np.array(self.state[(number_of_total_stocks+1):(number_of_total_stocks*2+1)]))

            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self.sell(index, actions[index])

            for index in buy_index:
                self.buy(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         

            self.state =  [self.state[0]] + \
                    self.data.AdjClose.values.tolist() + \
                    list(self.state[(number_of_total_stocks+1):(number_of_total_stocks*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist() + \
                    self.data.cci.values.tolist() + \
                    self.data.adx.values.tolist()
            
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(number_of_total_stocks+1)])*np.array(self.state[(number_of_total_stocks+1):(number_of_total_stocks*2+1)]))
            self.asset_memory.append(end_total_asset) 
            self.reward = end_total_asset - begin_total_asset            
            self.rewards_memory.append(self.reward)
            self.reward = self.reward*reward_scaler

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [start_balance]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        self.state = [start_balance] + \
                      self.data.AdjClose.values.tolist() + \
                      [0]*number_of_total_stocks + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist() 
        return self.state
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]