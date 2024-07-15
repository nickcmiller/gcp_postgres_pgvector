import logging
import sqlalchemy
from sqlalchemy.pool import QueuePool
from typing import Any, Callable
from google.cloud.sql.connector import Connector
from contextlib import contextmanager

from cloud_sql_gcp.config.gcp_sql_config import Config
from cloud_sql_gcp.utils.logging import setup_logging

logger = setup_logging()

def create_connector() -> Any:
    logger.info("Creating Google Cloud SQL connector")
    return Connector()

def get_connection(
    config: Config, 
    connector: Any
) -> Callable[[], Any]:
    logger.info("Creating database connection function")
    def getconn():
        logger.debug("Establishing database connection")
        conn = connector.connect(
            instance_connection_string=config.SQL_HOST,
            user=config.SQL_USER,
            password=config.SQL_PASSWORD,
            db=config.SQL_DATABASE,
            driver="pg8000"
        )
        logger.debug("Database connection established")
        return conn
    return getconn

def create_engine(
    getconn: Callable[[], Any]
) -> Any:
    logger.info("Creating SQLAlchemy engine with connection pooling")
    engine = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800
    )
    logger.debug("SQLAlchemy engine with connection pooling created")
    return engine

@contextmanager
def get_db_engine(config: Config):
    connector = create_connector()
    try:
        getconn = get_connection(config, connector)
        engine = create_engine(getconn)
        yield engine
    finally:
        connector.close()

@contextmanager
def get_db_connection(engine):
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()