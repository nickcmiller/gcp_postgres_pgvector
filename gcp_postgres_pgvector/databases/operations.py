from typing import Any, Dict, List, Optional
import numpy as np
import json

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pgvector.sqlalchemy import Vector
import sqlalchemy
from sqlalchemy import inspect, text, select, func
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import insert

from gcp_postgres_pgvector.utils.logging import setup_logging
from gcp_postgres_pgvector.utils.serialization import serialize_complex_types, deserialize_complex_types
from gcp_postgres_pgvector.utils.retry import db_retry_decorator

logger = setup_logging()

@db_retry_decorator()
def ensure_pgvector_extension(engine: Any) -> None:
    logger.info("Ensuring pgvector extension is enabled")
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
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
    engine: Any,
    table_name: str,
    query_embedding: List[float],
    limit: int = 5,
    where_clause: str = None,
    include_embedding: bool = False,
    included_columns: List[str] = ["id", "text", "title", "start_mins"]
) -> List[Dict[str, Any]]:
    logger.info(f"Reading similar rows from table '{table_name}'")
    try:
        with engine.connect() as connection:
            # Construct the base query
            query = "SELECT id, 1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity"
            for column in included_columns:
                query += f", {column}"

            if include_embedding:
                query += ", embedding"
            
            query += f" FROM {table_name}"
            
            if where_clause:
                query += f" WHERE {where_clause}"
            
            query += " ORDER BY similarity DESC LIMIT :limit"
            
            # Execute the query
            result = connection.execute(
                text(query),
                {"query_embedding": json.dumps(query_embedding), "limit": limit}
            )
            
            # Fetch and process the results
            rows = result.fetchall()
            results = []
            for row in rows:
                item = dict(row._mapping)  # Convert Row object to dictionary
                item['similarity'] = item.pop('similarity')  # Move similarity to the end
                if not include_embedding:
                    item.pop('embedding', None)
                results.append(item)
            
            return results
    except Exception as e:
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