import pandas as pd
import os
try:
    import pysqlite3 as sqlite3
    print("Using pysqlite3-binary.")
except ImportError:
    import sqlite3
    print("Using standard sqlite3.")


# --- Configuration ---
DATABASE_NAME = 'database.db'
# Replace with the actual paths to your CSV files
CSV_FILES_INFO = [
    {'path': 'Music Info.csv', 'table_name': 'songs_data'},
    {'path': 'User Listening History.csv', 'table_name': 'interactions_data'}
]

# --- Database Creation and Data Loading ---
def create_and_populate_db():
    # Remove existing database file if it exists to start fresh
    if os.path.exists(DATABASE_NAME):
        os.remove(DATABASE_NAME)
        print(f"Removed existing database: {DATABASE_NAME}")

    conn = None  # Initialize conn to None
    try:
        # Create a connection to the SQLite database
        # This will create the database file if it doesn't exist
        conn = sqlite3.connect(DATABASE_NAME)
        print(f"Database '{DATABASE_NAME}' created successfully.")

        # Load data from each CSV into a new table
        for file_info in CSV_FILES_INFO:
            csv_path = file_info['path']
            table_name = file_info['table_name']

            print(f"Processing {csv_path} into table '{table_name}'...")

            # For very large files, read in chunks
            # Adjust chunksize based on your system's memory
            chunk_size = 100000
            first_chunk = True
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                # In the first chunk, create the table (or replace if it exists)
                # For subsequent chunks, append the data
                if_exists_action = 'replace' if first_chunk else 'append'
                chunk.to_sql(table_name, conn, if_exists=if_exists_action, index=False)
                first_chunk = False
                print(f"  Loaded a chunk into '{table_name}'")

            print(f"Successfully loaded all data from {csv_path} into table '{table_name}'.")

        print("Database population complete.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except pd.errors.EmptyDataError:
        print(f"Error: One of the CSV files is empty or not found at the specified path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print(f"Database connection closed.")

if __name__ == '__main__':
    create_and_populate_db()