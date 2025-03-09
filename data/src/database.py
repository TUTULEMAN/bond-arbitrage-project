from psycopg import connect, Connection
from time import time
import psycopg as pg

class Database:
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection_time = -1 

    def connect(self) -> Connection:
        return connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )
    
    def create_tables(self):
        with self.connect() as conn:
            with conn.cursor() as cur:
                with open('./sql/CREATE_TABLES.sql') as f:
                    cur.execute(f.read())