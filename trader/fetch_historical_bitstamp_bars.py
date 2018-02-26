import quandl
from utils import config
from utils.db import pandas_to_db


def bitcoin_daily():
    """
    Fetches historical daily bars from Bitstamp exchange, with Quandl API
    :return: DB table with daily data formated on hourly level (missing hourly data)
    """
    quandl.ApiConfig.api_key = config.api_key
    bitstamp = quandl.get('BCHARTS/BITSTAMPUSD')
    bitstamp = bitstamp.asfreq('H')

    return pandas_to_db(bitstamp, db_name="trader", table_name="bitstamp_ohlcv")


if __name__ == "__main__":
    bitcoin_daily()
