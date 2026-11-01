import os
import psycopg2
from psycopg2 import OperationalError


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "traffic_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "port": int(os.getenv("DB_PORT", "5432"))
}


def get_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except OperationalError as error:
        print(f"Database connection failed: {error}")
        return None


def test_connection():
    connection = get_connection()

    if connection is None:
        return False

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), version()")
            result = cursor.fetchone()

        print("PostgreSQL connected successfully!")
        print("Database:", result[0])
        print("Version:", result[1])
        return True
    finally:
        connection.close()


if __name__ == "__main__":
    test_connection()
