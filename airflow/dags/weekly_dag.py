from datetime import timedelta
import airflow
from airflow.operators.bash_operator import BashOperator
from airflow.models import DAG

default_args = {
    'owner': 'SmartCat',
    'start_date': airflow.utils.dates.days_ago(8),
    'retries': 1,
    'retry_delay': timedelta(minutes=10)
}

dag = DAG(
    dag_id='weekly_trading_updates',
    default_args=default_args,
    schedule_interval='@weekly'
)

fetch_bitstamp = BashOperator(
    task_id='fetch_bitstamp',
    bash_command='docker exec trader_container bash -c "python fetch_historical_bitstamp_bars.py"',
    dag=dag
)

kill_running_trader = BashOperator(
    task_id='kill_running_trader',
    bash_command='docker exec trader_container bash -c "ps aux | grep \"python RLtrader.py\" | awk \"{print \$2}\" | xargs kill"',
    dag=dag
)

backtest_RL_and_trade = BashOperator(
    task_id='backtest_RL_and_trade',
    bash_command='docker exec -d trader_container bash -c "python RLtrader.py"',
    dag=dag
)

fetch_bitstamp.set_downstream(kill_running_trader)
kill_running_trader.set_downstream(backtest_RL_and_trade)
