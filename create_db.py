import pandas as pd
import os
import json
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi

try:
    import pysqlite3 as sqlite3
    print("Using pysqlite3-binary.")
except ImportError:
    import sqlite3
    print("Using standard sqlite3.")

# --- Configuration ---
DATABASE_NAME = 'database.db'

# Kaggle dataset information
KAGGLE_DATASET = "undefinenull/million-song-dataset-spotify-lastfm"  # Replace with your actual dataset
CSV_FILES_INFO = [
    {'filename': 'Music Info.csv', 'table_name': 'songs_data'},
    {'filename': 'User Listening History.csv', 'table_name': 'interactions_data'}
]

def setup_kaggle_credentials():
    """Set up Kaggle credentials from Streamlit secrets"""
    try:
        if not os.path.exists(os.path.expanduser('~/.kaggle')):
            os.makedirs(os.path.expanduser('~/.kaggle'))
        
        # Get Kaggle credentials from Streamlit secrets
        if hasattr(st, 'secrets') and 'kaggle' in st.secrets:
            kaggle_token = {
                'username': st.secrets['kaggle']['username'],
                'key': st.secrets['kaggle']['key']
            }
            
            # Save credentials to kaggle.json
            with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
                json.dump(kaggle_token, f)
            
            # Set appropriate permissions
            os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
            return True
        else:
            print("Kaggle credentials not found in Streamlit secrets")
            return False
    except Exception as e:
        print(f"Error setting up Kaggle credentials: {e}")
        return False

def download_kaggle_dataset():
    """Download dataset from Kaggle"""
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Download the dataset
        api.dataset_download_files(KAGGLE_DATASET, path='.', unzip=True)
        return True
    except Exception as e:
        print(f"Error downloading Kaggle dataset: {e}")
        if hasattr(st, 'error'):
            st.error(f"Error downloading Kaggle dataset: {e}")
        return False

# --- Database Creation and Data Loading ---
def create_and_populate_db():
    # Set up Kaggle credentials if we're in Streamlit Cloud
    if hasattr(st, 'secrets'):
        if not setup_kaggle_credentials():
            raise Exception("Failed to set up Kaggle credentials")
        
        # Download dataset from Kaggle
        if not download_kaggle_dataset():
            raise Exception("Failed to download Kaggle dataset")
    
    # Remove existing database file if it exists to start fresh
    if os.path.exists(DATABASE_NAME):
        os.remove(DATABASE_NAME)
        print(f"Removed existing database: {DATABASE_NAME}")

    conn = None
    try:
        # Create a connection to the SQLite database
        conn = sqlite3.connect(DATABASE_NAME)
        print(f"Database '{DATABASE_NAME}' created successfully.")

        # Load data from each CSV into a new table
        for file_info in CSV_FILES_INFO:
            filename = file_info['filename']
            table_name = file_info['table_name']

            if not os.path.exists(filename):
                raise FileNotFoundError(f"CSV file not found: {filename}")

            print(f"Processing {filename} into table '{table_name}'...")

            # For very large files, read in chunks
            chunk_size = 100000
            first_chunk = True
            for chunk in pd.read_csv(filename, chunksize=chunk_size):
                if_exists_action = 'replace' if first_chunk else 'append'
                chunk.to_sql(table_name, conn, if_exists=if_exists_action, index=False)
                first_chunk = False
                print(f"  Loaded a chunk into '{table_name}'")

            print(f"Successfully loaded all data from {filename} into table '{table_name}'.")

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