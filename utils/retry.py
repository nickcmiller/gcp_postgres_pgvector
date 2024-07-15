from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import sqlalchemy

def db_retry_decorator(
    max_attempts=3,
    wait_exponential_multiplier=1,
    wait_exponential_min=4,
    wait_exponential_max=10,
):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=wait_exponential_multiplier, min=wait_exponential_min, max=wait_exponential_max),
        retry=retry_if_exception_type((sqlalchemy.exc.OperationalError, sqlalchemy.exc.DatabaseError))
    )