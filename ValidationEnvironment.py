import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from environment_config import *



class StockValidation(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, day = 0, turbulence_threshold=140, iteration=''):

        self.day = day
        self.df = df
        self.action_space = spaces.Box(low = -1, high = 1,shape = (number_of_total_stocks,)) 
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (13,))
        self.data = self.df.loc[self.day,:]
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold
        self.state = [start_balance] + \
                      self.data.AdjClose.values.tolist() + \
                      [0]*number_of_total_stocks + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.asset_memory = [start_balance]
        self.rewards_memory = []
        self._seed()
        
        self.iteration=iteration


    def sell(self, index, action):

        macd = self.state[index+number_of_total_stocks+1]
        rsi = self.state[index+2*number_of_total_stocks]+1
        cci = self.state[index+3*number_of_total_stocks]+1
        adx = self.state[index+4*number_of_total_stocks]+1

        if self.turbulence<self.turbulence_threshold:
            if (cci < -100 or rsi) > 70 and adx > 50:
                if self.state[index+number_of_total_stocks+1] > 0:
                    self.state[0] += \
                    self.state[index+1]*min(abs(action),self.state[index+number_of_total_stocks+1]) 
                    
                    self.state[index+number_of_total_stocks+1] -= min(abs(action), self.state[index+number_of_total_stocks+1])
                    self.cost +=self.state[index+1]*min(abs(action),self.state[index+number_of_total_stocks+1])
                    self.trades+=1
            else:
                pass
        else:
            if self.state[index+number_of_total_stocks+1] > 0:
                self.state[0] += self.state[index+1]*self.state[index+number_of_total_stocks+1]
                self.state[index+number_of_total_stocks+1] =0
                self.cost += self.state[index+1]*self.state[index+number_of_total_stocks+1]
                self.trades+=1
            else:
                pass
    
    def buy(self, index, action):
        macd = self.state[index+number_of_total_stocks+1]
        rsi = self.state[index+2*number_of_total_stocks]+1
        cci = self.state[index+3*number_of_total_stocks]+1
        adx = self.state[index+4*number_of_total_stocks]+1
        if (cci > 100 or rsi < 30) and adx > 50:
            if self.turbulence< self.turbulence_threshold:
                available_amount = self.state[0] // self.state[index+1]
                self.state[0] -= self.state[index+1]*min(available_amount, action)
                self.state[index+number_of_total_stocks+1] += min(available_amount, action)
                self.cost+=self.state[index+1]*min(available_amount, action)
                self.trades+=1
        else:
            pass
        
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            plt.plot(self.asset_memory,'r')
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/validation_{}.csv'.format(self.iteration))
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(number_of_total_stocks+1)])*np.array(self.state[(number_of_total_stocks+1):(number_of_total_stocks*2+1)]))


            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)

            return self.state, self.reward, self.terminal,{}

        else:


            actions = actions * total_share
  
            if self.turbulence>=self.turbulence_threshold:
                actions=np.array([-total_share]*number_of_total_stocks)
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(number_of_total_stocks+1)])*np.array(self.state[(number_of_total_stocks+1):(number_of_total_stocks*2+1)]))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self.sell(index, actions[index])

            for index in buy_index:
                self.sell(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            self.turbulence = self.data['turbulence'].values[0]
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
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        self.state = [start_balance] + \
                      self.data.AdjClose.values.tolist() + \
                      [0]*number_of_total_stocks + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist()  + \
                      self.data.cci.values.tolist()  + \
                      self.data.adx.values.tolist() 
            
        return self.state
    
    def render(self, mode='human',close=False):
        return self.state
    

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]