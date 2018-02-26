import numpy as np
import tensorflow as tf
import random
import pandas as pd
import datetime, os
from utils.db import db_to_pandas, pandas_to_db, make_con
from pyalgotrade import bar, strategy
from pyalgotrade.technical import ma
from pyalgotrade.bitstamp import broker
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.bitstamp import barfeed

N_SIM = 100
INITIAL_CASH = 1000.0
# how large of a window of BTC prices to view
# ( 21/mth, 63/qtr, 251/yr for backtesting, number of order_book updates for paper trading)
HISTORY = 300


class QLearningDecisionPolicy():
    def __init__(self, actions, n_input, simnum, pretrained=False):
        self.model_path = "data/model_" + str(simnum)

        self.epsilon = 0.9      # how frequently to try a random action 1-epsilon == random %
        self.gamma = 0.001      # how far back to remember
        self.actions = actions

        n_output = len(actions)
        n_hidden = n_input - 2      # budget and n_coins are tacked onto end of input

        if pretrained:
            self.sess = tf.Session()

            new_saver = tf.train.import_meta_graph('data/model_' + str(simnum) + '/model.meta')
            new_saver.restore(self.sess, tf.train.latest_checkpoint('data/model_' + str(simnum) + '/'))

            self.x = tf.get_default_graph().get_tensor_by_name('x:0')
            self.y = tf.get_default_graph().get_tensor_by_name('y:0')

            W1 = tf.get_variable(name='W1', shape=[n_input, n_hidden])
            b1 = tf.get_variable(name='b1', shape=[n_hidden])
            h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
            W2 = tf.get_variable(name='W2', shape=[n_hidden, n_output])
            b2 = tf.get_variable(name='b2', shape=[n_output])

            self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)

            loss = tf.square(self.y - self.q)
            self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

            self.sess.run(tf.global_variables_initializer())
            # print("b1 : %s" % b1.eval(session=self.sess))
        else:
            self.x = tf.placeholder(tf.float32, [None, n_input], name='x')
            self.y = tf.placeholder(tf.float32, [n_output], name='y')

            W1 = tf.Variable(tf.random_normal([n_input, n_hidden]), name='W1')
            b1 = tf.Variable(tf.constant(0.1, shape=[n_hidden]), name='b1')
            h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
            W2 = tf.Variable(tf.random_normal([n_hidden, n_output]), name='W2')
            b2 = tf.Variable(tf.constant(0.1, shape=[n_output]), name='b2')

            self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)

            loss = tf.square(self.y - self.q)
            self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def select_action(self, current_state): #, step):

        # threshold = min(self.epsilon, step / 1000.)

        # if random number (0-1) > epsilon .9 try a random move ~10%
        if random.random() < self.epsilon:   # threshold:     # take best known action

            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            action_idx = np.argmax(action_q_vals)
            action = self.actions[action_idx]

        else:                               # random
            action = self.actions[random.randint(0, len(self.actions) - 1)]

        return action

    def update_q(self, state, reward, next_state):

        action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})

        next_action_idx = np.argmax(next_action_q_vals)
        action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = np.squeeze(np.asarray(action_q_vals))

        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})

    def save_sess(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.saver.save(self.sess, self.model_path + "/model")


class RLTraderBacktesting(strategy.BacktestingStrategy):
    def __init__(self, feed, *params):
        actions, initialCash, ncoins, history, simnum = params
        policy = QLearningDecisionPolicy(actions, history + 2, simnum)
        brk = broker.BacktestingBroker(initialCash, feed)

        strategy.BacktestingStrategy.__init__(self, feed, brk)

        self.__instrument = "BTC"
        self.__prices = feed[self.__instrument].getCloseDataSeries()
        self.__position = None
        self.__posSize = 0.05

        self.__RLinitialCash = initialCash
        self.__RLncoins = ncoins
        self.__RLpolicy = policy
        self.__RLhistory = history

    def onEnterOk(self, position):
        self.info("Position opened at %s" % (position.getEntryOrder().getExecutionInfo().getPrice()))

    def onEnterCanceled(self, position):
        self.info("Position entry canceled")
        self.__position = None

    def onExitOk(self, position):
        self.__position = None
        self.info("Position closed at %s" % (position.getExitOrder().getExecutionInfo().getPrice()))

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        self.__position.exitLimit(position.getExitOrder().getExecutionInfo().getPrice())

    def onBars(self, bars):
        # get Close price and budget ...
        bar = bars[self.__instrument]
        price = bar.getClose()

        # history prices
        prices = self.getFeed()[self.__instrument].getCloseDataSeries()[-self.__RLhistory:]
        # if list is shorter than history, prepend zeros
        target_length = self.__RLhistory
        if len(prices) < target_length:
            prices_lst = [0] * (target_length - len(prices)) + prices
        else:
            prices_lst = prices

        # array shape manipulations
        budget = np.array([self.__RLinitialCash]).reshape(1, 1)
        n_coins = np.array([self.__RLncoins]).reshape(1, 1)
        i_prices = np.array(prices_lst).reshape(1, self.__RLhistory)

        current_state = np.asmatrix(np.hstack((i_prices, budget, n_coins)))
        current_portfolio = self.getBroker().getEquity()
        action = self.__RLpolicy.select_action(current_state)

        # If a position was not opened, check if policy gave "Buy" action and enter a long position.
        if self.__position is None:
            if action == "Buy":
                self.info("Policy action 'BUY'. Buy at %s" % price)
                try:
                    self.__position = self.enterLongLimit(
                        self.__instrument, price, self.__posSize, True)
                    self.__RLncoins += self.__posSize
                except Exception as e:
                    self.error("Failed to buy: %s" % e)
            else:
                pass
        # Check if we have to close the position.
        elif not self.__position.exitActive() and action == "Sell":
            try:
                self.info("Policy action 'SELL'. Sell at %s" % price)
                self.__position.exitLimit(price)
                self.__RLncoins -= self.__posSize
            except Exception as e:
                self.error("Failed to sell: %s" % e)
        else:
            pass

        new_portfolio = self.getBroker().getCash() + self.__RLncoins * price
        reward = new_portfolio - current_portfolio

        n_coins = np.array([self.__RLncoins]).reshape(1, 1)
        next_state = np.asmatrix(np.hstack((i_prices, budget, n_coins)))

        self.__RLpolicy.update_q(current_state, reward, next_state)

    def onFinish(self, bars):
        self.__RLpolicy.save_sess()


class RLTraderPaperTrade(strategy.BaseStrategy):
    def __init__(self, feed, brk, actions, history, initialCash, ncoins, best_model, start_time):
        policy = QLearningDecisionPolicy(actions, history + 2, simnum=best_model, pretrained=True)

        strategy.BaseStrategy.__init__(self, feed, brk)
        self.__instrument = "BTC"
        self.__prices = feed[self.__instrument].getCloseDataSeries()
        self.__sma = ma.SMA(self.__prices, 15)
        self.__position = None
        self.__bid = None
        self.__ask = None
        self.__posSize = 0.05

        self.__RLinitialCash = initialCash
        self.__RLncoins = ncoins
        self.__RLpolicy = policy
        self.__RLhistory = history
        self.__start_time = start_time

        # Subscribe to order book update events to get bid/ask prices to trade.
        feed.getOrderBookUpdateEvent().subscribe(self.__onOrderBookUpdate)

    def __onOrderBookUpdate(self, orderBookUpdate):
        bid = orderBookUpdate.getBidPrices()[0]
        ask = orderBookUpdate.getAskPrices()[0]

        if bid != self.__bid or ask != self.__ask:
            self.__bid = bid
            self.__ask = ask
            self.info("Order book updated. Best bid: %s. Best ask: %s" % (self.__bid, self.__ask))

    def getMA(self):
        return self.__sma

    def onEnterOk(self, position):
        self.info("Position opened at %s" % (position.getEntryOrder().getExecutionInfo().getPrice()))

    def onEnterCanceled(self, position):
        self.info("Position entry canceled")
        self.__position = None

    def onExitOk(self, position):
        self.__position = None
        self.info("Position closed at %s" % (position.getExitOrder().getExecutionInfo().getPrice()))

    def onExitCanceled(self, position):
        self.__position.exitLimit(position.getExitOrder().getExecutionInfo().getPrice())

    def onBars(self, bars):
        # get Close price and budget ...
        bar = bars[self.__instrument]
        price = bar.getClose()

        # history prices
        prices = self.getFeed()[self.__instrument].getCloseDataSeries()[-self.__RLhistory:]
        # if list is shorter than history, prepend zeros
        target_length = self.__RLhistory
        if len(prices) < target_length:
            prices_lst = [0] * (target_length - len(prices)) + prices
        else:
            prices_lst = prices

        # array shape manipulations
        budget = np.array([self.__RLinitialCash]).reshape(1, 1)
        n_coins = np.array([self.__RLncoins]).reshape(1, 1)
        i_prices = np.array(prices_lst).reshape(1, self.__RLhistory)

        current_state = np.asmatrix(np.hstack((i_prices, budget, n_coins)))
        current_portfolio = self.getBroker().getEquity()
        action = self.__RLpolicy.select_action(current_state)

        # If a position was not opened, check if policy gave "Buy" action and enter a long position.
        if self.__position is None:
            if action == "Buy":
                self.info("Policy action 'BUY'. Trying to Buy at %s" % self.__ask)
                try:
                    self.__position = self.enterLongLimit(
                        self.__instrument, self.__ask, self.__posSize, True)
                    self.__RLncoins += self.__posSize
                    # update DB
                    update_df = pd.DataFrame({"Start time": [self.__start_time],
                                  "Action": [action],
                                  "Portfolio value": [self.getResult()]},
                                 index=[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                    pandas_to_db(update_df, "trader", "rl_papertrade")
                except Exception as e:
                    self.error("Failed to buy: %s" % e)
            else:
                pass
        # Check if we have to close the position.
        elif not self.__position.exitActive() and action == "Sell":
            try:
                self.info("Policy action 'SELL'. Trying to Sell at %s" % self.__bid)
                self.__position.exitLimit(self.__bid)
                self.__RLncoins -= self.__posSize
                # update DB
                update_df = pd.DataFrame({"Start time": [self.__start_time],
                                          "Action": [action],
                                          "Portfolio value": [self.getResult()]},
                                         index=[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                pandas_to_db(update_df, "trader", "rl_papertrade")
            except Exception as e:
                self.error("Failed to sell: %s" % e)
        else:
            pass

        new_portfolio = self.getBroker().getCash() + self.__RLncoins * price
        reward = new_portfolio - current_portfolio

        n_coins = np.array([self.__RLncoins]).reshape(1, 1)
        next_state = np.asmatrix(np.hstack((i_prices, budget, n_coins)))

        self.__RLpolicy.update_q(current_state, reward, next_state)


def backtest(nsim=N_SIM, history=HISTORY):
    """
    Backtest RL Trader on whole bitstamp dataset
    :param nsim: Number of simulation to run
    :return: New entry to DB with info about simulations
    """

    # prepare CSV to read bars from
    bars_df = db_to_pandas(db_name="trader", table_name="bitstamp_ohlcv", index_name="Datetime")
    bars_df.dropna(axis=0, inplace=True)
    bars_df.reset_index(inplace=True)
    bars_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "to_drop", "Adj Close"]
    bars_df.drop(axis=1, columns=["to_drop"], inplace=True)
    bars_df['Date'] = pd.to_datetime(bars_df['Date'], format='%Y-%m-%d')

    final_portfolios = list()

    for i in range(nsim):
        # read bars
        bars = quandlfeed.Feed(bar.Frequency.DAY)
        bars_df.to_csv("data/btc_bars.csv", index=False)
        bars.addBarsFromCSV('BTC', "data/btc_bars.csv")

        actions = ['Buy', 'Sell', 'Hold']
        budget = INITIAL_CASH       # initial cash on hand
        ncoins = 0.0
        params = [actions, budget, ncoins, history, i]

        RLstrat = RLTraderBacktesting(bars, *params)
        RLstrat.run()
        final_portfolios.append(RLstrat.getResult())

    # print "Average return out of {} backtesting simulations: {}".format(len(final_portfolios), np.mean(final_portfolios))
    # print "Standard deviation of return: {}".format(np.std(final_portfolios))
    # print "Best model with saved Q values: {}".format(np.argmax(final_portfolios))

    simulation_df = pd.DataFrame({"Number of simulations": [len(final_portfolios)],
                                  "Average return": [np.mean(final_portfolios)],
                                  "Std. deviation of return": [np.std(final_portfolios)],
                                  "Best model index": [np.argmax(final_portfolios)],
                                  "Best model return": [final_portfolios[np.argmax(final_portfolios)]]},
                                 index=[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    pandas_to_db(simulation_df, db_name="trader", table_name="rl_backtesting")


def paper_trade(history=HISTORY):
    """
    Live (paper) trade with RL Trader on Bitstamp,
    with best_model params from backtesting simulations
    :return: Update DB every on every Buy/Sell event
    """
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:00:00')
    actions = ['Buy', 'Sell', 'Hold']
    ncoins = 0.0
    bars = barfeed.LiveTradeFeed()
    brk = broker.PaperTradingBroker(INITIAL_CASH, bars)

    engine, con = make_con(db_name="trader")
    sql = "SELECT * FROM rl_backtesting ORDER BY Datetime DESC LIMIT 1"
    latest_simulation = pd.read_sql(sql, con=con, parse_dates=True, index_col='Datetime')
    best_model = latest_simulation['Best model index'].iloc[0]
    RLstrat = RLTraderPaperTrade(bars, brk, actions, history, INITIAL_CASH, ncoins, best_model, start_time)

    RLstrat.run()


if __name__ == "__main__":
    backtest()
    paper_trade()