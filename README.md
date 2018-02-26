# Superset Trading Dashboard

This is an experimental setup with simple goal to tryout 
**Apache Superset** as Trading dashboard + simple implementation of
**Reinforcement learning** approach for algo-trading with **Bitcoin**.
If you have an idea for how to add more hype to the pile, open an issue! 
  
Setup consists of 3 main modules:
- `trader`: Trading scripts for fetching historical Bitcoin prices and backtesting 
and live (paper) trading on Bitstamp, with simple Q-table RL algo-trader implemented 
with `tensorflow` and `pyalgotrade`:
$$\mathbf{b}$$
- `airflow`: Apache Airflow DAGs for scheduling initial and weekly tasks.
- `superset`: Apache Superset Dashboard 

## Building

First, make `superset-config.env` and `trader-config.env` files based on examples and populate keys 
(no spaces around = sign and no quotes).  

Build docker images:

```
docker-compose build
```

## Running

```
docker-compose up
```

or 

```
docker-compose up --build
```

## Make/Load Dashboard

In order to make or load Dashboard you'll need to prepare database, 
load it to Apache Superset and import Dashboard pickle file. 

- Go to Airflow UI (`localhost:8090`) and turn-on `weekly` DAG. 
- After DAG finishes, go to Superset UI (`localhost:8088`)
and login with credentials from env file. 
- Import database: internally it is mounted to `/etc/superset/db` and called `trader.db`.
- Import pickled Dashboard from superset/dashboard directory.
- Make some changes (add other traders, tweak RLtrader, etc..)
- Deploy somewhere, make guest user and invite people to show off your Dashboards!
