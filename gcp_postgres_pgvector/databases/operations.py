from typing import Any, Dict, List, Optional
import numpy as np
import json

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pgvector.sqlalchemy import Vector
import sqlalchemy
from sqlalchemy import inspect, text, select, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError

from gcp_postgres_pgvector.utils.logging import setup_logging
from gcp_postgres_pgvector.utils.serialization import serialize_complex_types, deserialize_complex_types
from gcp_postgres_pgvector.utils.retry import db_retry_decorator

logger = setup_logging()

@db_retry_decorator()
def ensure_pgvector_extension(
    engine: Any
) -> None:
    """
        Ensures that the pgvector extension is enabled in the PostgreSQL database.

        This function attempts to create the pgvector extension if it does not already exist.
        It uses a database connection to execute the necessary SQL commands. If the extension
        is successfully created or already exists, a log message is generated to indicate success.
        If there is an error during the process, an error message is logged, and an exception is raised.

        Args:
            engine (Any): The SQLAlchemy engine instance used to connect to the PostgreSQL database.

        Raises:
            Exception: If the extension creation fails or if there is an error during the execution
                    of the SQL commands.
    """
    logger.info("Ensuring pgvector extension is enabled")
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
            if result.fetchone() is None:
                raise Exception("Failed to create pgvector extension")
        logger.info("pgvector extension enabled successfully")
    except Exception as e:
        logger.error(f"Error ensuring pgvector extension: {e}", exc_info=True)
        raise

def ensure_table_schema(
    engine: Any, 
    table_name: str, 
    data_object: Dict[str, Any],
    unique_column: str = 'id',
    vector_dimensions: int = 3072,
    make_primary_key: bool = False
) -> None:
    """
        Ensures that the specified table schema exists in the PostgreSQL database.

        This function checks if a table with the given name already exists in the database.
        If it does, it verifies and ensures that the necessary constraints are applied.
        If the table does not exist, it creates a new table with the specified columns and constraints.

        Args:
            engine (Any): The SQLAlchemy engine instance used to connect to the PostgreSQL database.
            table_name (str): The name of the table to ensure exists in the database.
            data_object (Dict[str, Any]): A dictionary representing the columns and their data types for the table.
            unique_column (str, optional): The column to be used as a unique identifier for the table. Defaults to 'id'.
            vector_dimensions (int, optional): The number of dimensions for the vector type. Defaults to 3072.
            make_primary_key (bool, optional): A flag indicating whether to make the unique_column a primary key. Defaults to False.

        Raises:
            ValueError: If data_object is None or empty.
            Exception: If there is an error during the execution of SQL commands to create the table or apply constraints.

        Notes:
            - The function uses the SQLAlchemy `inspect` module to check for existing tables and constraints.
            - The `_generate_column_definitions` helper function is used to create the SQL column definitions based on the provided data_object.
            - The function logs the creation of the table and any errors encountered during the process.
    """
    if not data_object:
        raise ValueError("data_object cannot be None or empty")
    
    inspector = inspect(engine)
    if inspector.has_table(table_name):
        _ensure_constraints(engine, inspector, table_name, unique_column, make_primary_key)
        return

    columns = _generate_column_definitions(data_object, vector_dimensions)
    constraint = f"PRIMARY KEY ({unique_column})" if make_primary_key else f"UNIQUE ({unique_column})"
    columns.append(constraint)
    
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE {table_name} ({', '.join(columns)})"))
    
    logger.info(f"Created table {table_name} with {constraint} on {unique_column}")

def _generate_column_definitions(
    data_object: Dict[str, Any], 
    vector_dimensions: int
) -> List[str]:
    """
        Generates SQL column definitions based on the provided data_object.

        This helper function takes a dictionary representing the columns and their corresponding data types
        and constructs a list of SQL column definitions suitable for creating a table in PostgreSQL.

        Args:
            data_object (Dict[str, Any]): A dictionary where keys are column names and values are the data types
                                           or sample values for those columns.
            vector_dimensions (int): The number of dimensions for the vector type, used when the column name is 'embedding'.

        Returns:
            List[str]: A list of strings, each representing a column definition in SQL syntax.

        Example:
            If data_object is {'name': 'John Doe', 'age': 30, 'embedding': [0.1, 0.2, 0.3]},
            the function will return:
            [
                'name TEXT',
                'age INTEGER',
                'embedding Vector(3)'
            ]
    """
    def get_sql_type(column: str, value: Any) -> str:
        if column == 'embedding':
            return f"Vector({len(value) if isinstance(value, list) else vector_dimensions})"
        elif isinstance(value, str):
            return "TEXT"
        elif isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "FLOAT"
        else:
            return "TEXT"

    return [
        f"{column} {get_sql_type(column, value)}"
        for column, value in data_object.items()
    ]

def _ensure_constraints(engine: Any, 
    inspector: Any, 
    table_name: str, 
    unique_column: str, 
    make_primary_key: bool
) -> None:
    """
        Ensures that the specified constraints are applied to the given table in the database.

        This function checks if the specified unique column is already a primary key or if it needs to be added as a unique constraint.
        If the table does not have the required constraints, it will attempt to add them.

        Args:
            engine (Any): The SQLAlchemy engine instance used to connect to the database.
            inspector (Any): The SQLAlchemy inspector instance used to retrieve table metadata.
            table_name (str): The name of the table to check and modify constraints for.
            unique_column (str): The name of the column that should be unique or a primary key.
            make_primary_key (bool): A flag indicating whether to make the unique column a primary key.

        Returns:
            None: This function does not return a value. It performs operations directly on the database.

        Raises:
            Warning: If an attempt is made to modify an existing table to make a column a primary key when it already has a primary key.
            Info: Logs information about the operations performed, including whether constraints were added or already exist.

        Example:
            If you have a table named 'users' and you want to ensure that the 'email' column is unique,
            you would call this function as follows:
            
            _ensure_constraints(engine, inspector, 'users', 'email', make_primary_key=False)
    """
    pk_constraint = inspector.get_pk_constraint(table_name)
    is_primary_key = pk_constraint and unique_column in pk_constraint['constrained_columns']
    
    if make_primary_key and not is_primary_key:
        logger.warning(f"Cannot modify existing table '{table_name}' to make '{unique_column}' the primary key.")
    elif not make_primary_key and not is_primary_key:
        constraints = inspector.get_unique_constraints(table_name)
        if not any(unique_column in constraint['column_names'] for constraint in constraints):
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table_name} ADD CONSTRAINT {table_name}_{unique_column}_key UNIQUE ({unique_column})"))
            logger.info(f"Added unique constraint on '{unique_column}' column for existing table '{table_name}'")
    
    logger.info(f"Table {table_name} already exists. Ensured necessary constraints.")

@db_retry_decorator()
def write_list_of_objects_to_table(
    engine: Any,
    table_name: str,
    data_list: List[Dict[str, Any]],
    unique_column: str = 'id',
    batch_size: int = 1000,
    make_primary_key: bool = False
) -> None:
    """
        Writes a list of objects to a specified table in the database, handling potential conflicts
        and ensuring the table schema is correct before insertion.

        This function takes a list of dictionaries, where each dictionary represents an object to be
        inserted into the database table. It supports batch insertion to improve performance and can
        handle unique constraints by updating existing records if a conflict occurs.

        Args:
            engine (Any): The SQLAlchemy engine instance used to connect to the database.
            table_name (str): The name of the table to which the objects will be written.
            data_list (List[Dict[str, Any]]): A list of dictionaries containing the data to be inserted.
            unique_column (str, optional): The name of the column that should be treated as unique. 
                                        Defaults to 'id'.
            batch_size (int, optional): The number of records to insert in a single batch. 
                                        Defaults to 1000.
            make_primary_key (bool, optional): A flag indicating whether to make the unique column 
                                                a primary key. Defaults to False.

        Returns:
            None: This function does not return a value. It performs operations directly on the database.

        Raises:
            Exception: If an error occurs during the writing process, an exception is raised with 
                    an error message.

        Example:
            To write a list of user objects to a 'users' table, you would call this function as follows:

            write_list_of_objects_to_table(engine, 'users', [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}])
    """
    if not data_list:
        logger.info(f"No data to write to table '{table_name}'")
        return

    logger.info(f"Writing {len(data_list)} objects to table '{table_name}'")

    try:
        with engine.begin() as connection:
            if data_list:
                ensure_table_schema(engine, table_name, data_list[0], unique_column, make_primary_key=make_primary_key)
            
            table = sqlalchemy.Table(table_name, sqlalchemy.MetaData(), autoload_with=engine)
            insert_stmt = insert(table)
            
            update_dict = {c.name: c for c in insert_stmt.excluded if c.name != unique_column}
            insert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=[unique_column],
                set_=update_dict
            )
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i+batch_size]
                serialized_batch = []
                for data_object in batch:
                    serialized_data = {
                        k: serialize_complex_types(v) if k != 'embedding' else v 
                        for k, v in data_object.items()
                    }
                    
                    if 'embedding' in serialized_data and isinstance(serialized_data['embedding'], np.ndarray):
                        serialized_data['embedding'] = serialized_data['embedding'].tolist()
                    
                    serialized_batch.append(serialized_data)
                
                connection.execute(insert_stmt, serialized_batch)
                logger.info(f"Successfully processed {len(serialized_batch)} rows for table '{table_name}'")
        
        logger.info(f"Finished writing all {len(data_list)} objects to table '{table_name}'")
    except Exception as e:
        logger.error(f"Error writing to table '{table_name}': {e}", exc_info=True)
        raise

@db_retry_decorator()
def read_from_table(
    engine: Any, 
    table_name: str,
    limit: int = None,
    where_clause: str = None,
    include_embedding: bool = False
) -> List[Dict[str, Any]]:
    logger.info(f"Reading data from table '{table_name}'")
    try:
        with engine.connect() as connection:
            query = select(text("*")).select_from(text(table_name))
            if where_clause:
                query = query.where(text(where_clause))
            if limit:
                query = query.limit(limit)
            
            result = connection.execute(query)
            rows = [row._asdict() for row in result]

        processed_rows = []
        for row in rows:
            processed_row = {}
            for key, value in row.items():
                if key != 'embedding' or include_embedding:
                    processed_row[key] = deserialize_complex_types(value)
            processed_rows.append(processed_row)

        logger.info(f"Successfully read {len(processed_rows)} rows from table '{table_name}'")
        return processed_rows
    except Exception as e:
        logger.error(f"Error reading from table '{table_name}': {e}", exc_info=True)
        raise

@db_retry_decorator()
def read_similar_rows(
    engine: Engine,
    table_name: str,
    query_embedding: List[float],
    limit: int = 5,
    where_clause: Optional[str] = None,
    include_embedding: bool = False,
    included_columns: List[str] = ["id", "text", "title", "start_mins"],
    similarity_threshold: float = 0.35
) -> List[Dict[str, Any]]:
    """
        Reads similar rows from the specified table in the PostgreSQL database based on a query embedding.

        This function calculates the similarity of the provided query embedding against the embeddings stored in the specified table.
        It retrieves rows that meet a defined similarity threshold and can include or exclude the embedding column in the results.

        Args:
            engine (Engine): The SQLAlchemy engine instance used to connect to the PostgreSQL database.
            table_name (str): The name of the table from which to read similar rows.
            query_embedding (List[float]): The embedding vector to compare against the stored embeddings.
            limit (int, optional): The maximum number of similar rows to return. Defaults to 5.
            where_clause (Optional[str], optional): An optional SQL WHERE clause to filter the results. Defaults to None.
            include_embedding (bool, optional): A flag indicating whether to include the embedding column in the results. Defaults to False.
            included_columns (List[str], optional): A list of column names to include in the results. Defaults to ["id", "text", "title", "start_mins"].
            similarity_threshold (float, optional): The minimum similarity score for a row to be included in the results. Defaults to 0.35.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a row of similar data from the table.

        Raises:
            SQLAlchemyError: If there is an error during the execution of the SQL commands to read similar rows.

        Example:
            To read similar rows from a table named 'documents' using a specific embedding:
            similar_rows = read_similar_rows(engine, 'documents', query_embedding=[0.1, 0.2, 0.3], limit=10)
    """
    logger.info(f"Reading similar rows from table '{table_name}'")
    try:
        with engine.connect() as connection:
            columns = ", ".join(included_columns)
            similarity_calc = "1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity"
            
            cte = f"""
            WITH similarity_cte AS (
                SELECT id, {similarity_calc}, {columns}{', embedding' if include_embedding else ''}
                FROM {table_name}
                {f'WHERE {where_clause}' if where_clause else ''}
            )
            SELECT *
            FROM similarity_cte
            WHERE similarity >= :similarity_threshold
            ORDER BY similarity DESC
            LIMIT :limit
            """
            
            result = connection.execute(
                text(cte),
                {
                    "query_embedding": json.dumps(query_embedding),
                    "limit": limit,
                    "similarity_threshold": similarity_threshold
                }
            )

            results = []
            for row in result:
                item = dict(row._mapping)  # Convert Row object to dictionary
                
                if not include_embedding:
                    item.pop('embedding', None)
                results.append(item)
            
            return results
    except SQLAlchemyError as e:
        logger.error(f"Error reading similar rows: {str(e)}")
        raise

@db_retry_decorator()
def delete_table(
    engine: Any, 
    table_name: str
) -> None:
    logger.info(f"Deleting table '{table_name}'")
    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        logger.info(f"Successfully deleted table '{table_name}'")
    except Exception as e:
        logger.error(f"Error deleting table '{table_name}': {e}", exc_info=True)
        raise