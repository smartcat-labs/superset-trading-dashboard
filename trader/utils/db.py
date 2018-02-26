import pandas as pd
from sqlalchemy import create_engine, inspect
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'sqlite'))


def make_con(db_name):
    """
    Returns connection to DB
    :param db_name: DB name
    :return: engine and connection
    """
    db = os.path.join(DB_PATH, db_name+'.db')

    if not os.path.exists(db):
        open(db, 'a').close()

    sql_engine = create_engine('sqlite:///'+db, echo=False)
    connection = sql_engine.raw_connection()

    return sql_engine, connection


def pandas_to_db(df, db_name, table_name, index_name='Datetime'):
    """
    Saves pandas DataFrame to DB conn
    :param df: pandas Dataframe, datetime index
    :param table: string, table name
    """
    # pd.DataFrame.to_sql doesn't have ON DUPLICATE KEY UPDATE or similar
    # so we will make temp table with to_sql and then merge
    engine, con = make_con(db_name=db_name)

    inspector = inspect(engine)
    df.index.name = index_name
    df.reset_index(inplace=True)
    df[index_name] = df[index_name].astype(str)

    if table_name in inspector.get_table_names():
        diff_df = clean_df_db_dups(df, table_name, engine, index_name)
        diff_df.to_sql(name=table_name, con=con, if_exists='append', index=False)
    else:
        df.to_sql(name=table_name, con=con, if_exists='append', index=False)


def clean_df_db_dups(df, table_name, engine, index_name):
    """
    Remove rows from a dataframe that already exist in a database
    :param df: df to check for duplicates
    :param table_name: table name to query from DB
    :param engine: engine from make_con()
    :param index_name: name of the index in DF
    :return: DF with unique records compared to DB table records
    """

    sql = 'SELECT {} FROM {}'.format(index_name, table_name)
    df.drop_duplicates(index_name, keep='last', inplace=True)

    df = pd.merge(df, pd.read_sql(sql, engine), how='left', on=index_name, indicator=True)
    df = df[df['_merge'] == 'left_only']
    df.drop(['_merge'], axis=1, inplace=True)

    return df


def db_to_pandas(db_name, table_name, index_name, index_as_dates=True):
    """
    Queries DB for all records from table
    :param db_name: Database file
    :param table_name: Name of table to return as DF
    :param index_name: Name of index in DB table
    :param index_as_dates: Bool, True if index should be parsed as Datetime index
    :return: pandas DF
    """

    engine, con = make_con(db_name=db_name)

    sql = "SELECT * FROM " + table_name

    return pd.read_sql(sql=sql, con=con, index_col=index_name, parse_dates=index_as_dates)
