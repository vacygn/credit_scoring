import pandas as pd
import psycopg2
from src.config.db_config import DB_ARGS


class PostgresConnector(object):
    def __init__(self, db_args=DB_ARGS):
        self._args = db_args

    def send_sql_query(self, query: str):
        conn = psycopg2.connect(**self._args)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)
        finally:
            if conn:
                cursor.close()
                conn.close()

    def get_df_from_query(self, query: str):
        conn = psycopg2.connect(**self._args)
        df = pd.read_sql(query, conn)
        conn.close()
        return df
