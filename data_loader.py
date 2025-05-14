import pandas as pd
import streamlit as st
try:
    import pysqlite3 as sqlite3
    print("Using pysqlite3-binary.")
except ImportError:
    import sqlite3
    print("Using standard sqlite3.")

# Configuration
DATABASE_NAME = 'database.db' # Must match the name used in create_db.py

#Schema for the SQLite database
# This schema should match the structure of your SQLite tables
SCHEMA = {
    "track_id": "track_id",
    "track_name": "name", 
    "artist": "artist",
    "tags": "tags",
    "year": "year",
    "features": ["danceability", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"],
    "playcount": "playcount",
    "user_id": "user_id",
    #these are the columns in your interactions_data table
    "user_idx": "user_idx",  
    "track_idx": "track_idx" 
}
def load_data():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        print(f"Successfully connected to database: {DATABASE_NAME}")

        # --- Load songs data ---
        # Adjust the SELECT query based on your 'songs_data' table structure
        # Ensure the column names here match those in your SQLite 'songs_data' table
        query_songs = "SELECT track_id, name, artist, tags, year, danceability, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo FROM songs_data" # Example Query
        df_songs = pd.read_sql_query(query_songs, conn)
        print(f"Loaded 'songs_data' into DataFrame with {len(df_songs)} rows.")


        # --- Load interactions data ---
        # Adjust the SELECT query based on your 'interactions_data' table structure
        # Ensure the column names here match those in your SQLite 'interactions_data' table
        query_interactions = "SELECT user_id, track_id, playcount FROM interactions_data" # Example Query
        df_inter = pd.read_sql_query(query_interactions, conn)
        print(f"Loaded 'interactions_data' into DataFrame with {len(df_inter)} rows.")


        # --- Data Preprocessing ---
        # Ensure the DataFrames have the expected columns

        # Create track_idx and user_idx if they are not directly in your tables
        # For df_songs:
        if 'track_idx' not in df_songs.columns:
            df_songs['track_idx'] = df_songs.index # Or a more sophisticated mapping if needed
        track2idx = pd.Series(df_songs.track_idx.values, index=df_songs[SCHEMA["track_id"]]).to_dict()
        idx2track = {v: k for k, v in track2idx.items()}

        # For df_inter:
        # Map track_id to track_idx
        df_inter['track_idx'] = df_inter[SCHEMA["track_id"]].map(track2idx)

        # Create user_idx if not in table
        if 'user_idx' not in df_inter.columns:
             # Create a unique mapping for user_id to a numerical user_idx
            user_ids = df_inter[SCHEMA["user_id"]].unique()
            user2idx = {user_id: i for i, user_id in enumerate(user_ids)}
            df_inter['user_idx'] = df_inter[SCHEMA["user_id"]].map(user2idx)

        # Drop rows with NaN in track_idx or user_idx if any were created due to missing mappings
        df_inter.dropna(subset=['track_idx', 'user_idx'], inplace=True)
        df_inter['user_idx'] = df_inter['user_idx'].astype(int)
        df_inter['track_idx'] = df_inter['track_idx'].astype(int)


        print("Data loading and preprocessing complete.")
        return df_songs, df_inter, SCHEMA, track2idx, idx2track

    except sqlite3.Error as e:
        st.error(f"SQLite error during data loading: {e}")
        print(f"SQLite error during data loading: {e}")
        # Return empty DataFrames or handle error as appropriate for your app
        return pd.DataFrame(), pd.DataFrame(), SCHEMA, {}, {}
    except KeyError as e:
        st.error(f"KeyError during data processing: {e}. Check your SCHEMA and SQL query column names.")
        print(f"KeyError during data processing: {e}. Check your SCHEMA and SQL query column names.")
        return pd.DataFrame(), pd.DataFrame(), SCHEMA, {}, {}
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        print(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame(), pd.DataFrame(), SCHEMA, {}, {}
    finally:
        if conn:
            conn.close()
            print("Database connection closed in load_data.")