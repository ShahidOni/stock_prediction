# common library
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv

from data_preprocessor import *
from models import *
import os
import tensorflow as tf
from data_preprocessor import *

tf.logging.set_verbosity(tf.logging.ERROR)
# import sys
# sys.stdout = open('/home/dmlab/Shain/rllog.txt', 'w')


def run_model() -> None:
    """Train the model."""

    data = process_data()
    data = add_turbulence(data)

    unique_trade_date = data[(data.Date > 20151001)&(data.Date <= 20211130)].Date.unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63
    

    run_(df=data, unique_trade_date= unique_trade_date, rebalance_window = rebalance_window, validation_window=validation_window)



if __name__ == "__main__":
    run_model()
    #sys.stdout.close()

