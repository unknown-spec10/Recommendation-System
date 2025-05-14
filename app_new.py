import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

# --- Configuration ---
DATABASE_NAME = 'database.db'

# Kaggle dataset information
KAGGLE_DATASET = "undefinenull/million-song-dataset-spotify-lastfm"  # Replace with your actual dataset
CSV_FILES_INFO = [
    {'filename': 'Music Info.csv', 'table_name': 'songs_data'},
    {'filename': 'User Listening History.csv', 'table_name': 'interactions_data'}
]

SCHEMA = {
    "track_id": "track_id",
    "track_name": "name",
    "artist": "artist",
    "tags": "tags",
    "year": "year",
    "features": ["danceability", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"],
    "playcount": "playcount",
    "user_id": "user_id",
    "user_idx": "user_idx",
    "track_idx": "track_idx"
}

def setup_kaggle_credentials():
    """Set up Kaggle credentials from Streamlit secrets"""
    try:
        # Try both possible locations
        kaggle_dirs = [
            os.path.expanduser('~/.kaggle'),
            os.path.expanduser('~/.config/kaggle')
        ]
        
        created = False
        for kaggle_dir in kaggle_dirs:
            if not os.path.exists(kaggle_dir):
                os.makedirs(kaggle_dir, exist_ok=True)
                created = True
        
        # Get Kaggle credentials from Streamlit secrets
        if hasattr(st, 'secrets') and 'kaggle' in st.secrets:
            kaggle_token = {
                'username': st.secrets['kaggle']['username'],
                'key': st.secrets['kaggle']['key']
            }
            
            # Save credentials to both possible locations
            for kaggle_dir in kaggle_dirs:
                kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
                with open(kaggle_json_path, 'w') as f:
                    json.dump(kaggle_token, f)
                os.chmod(kaggle_json_path, 0o600)
            
            # Set the KAGGLE_CONFIG_DIR environment variable to the first existing directory
            config_dir = next((d for d in kaggle_dirs if os.path.exists(d)), None)
            if config_dir:
                os.environ['KAGGLE_CONFIG_DIR'] = config_dir
                print(f"Set KAGGLE_CONFIG_DIR to {config_dir}")
            
            return True
        else:
            print("Kaggle credentials not found in Streamlit secrets")
            return False
    except Exception as e:
        print(f"Error setting up Kaggle credentials: {e}")
        return False

def get_kaggle_api():
    """Get a properly configured Kaggle API instance"""
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        
        # Set configuration
        config_path = os.environ.get('KAGGLE_CONFIG_DIR') or os.path.expanduser('~/.kaggle')
        if not os.path.exists(config_path):
            os.makedirs(config_path, exist_ok=True)
        
        # Try to authenticate
        try:
            api.authenticate()
        except Exception as e:
            print(f"Authentication failed: {e}")
            # Try to find the config file
            config_file = os.path.join(config_path, 'kaggle.json')
            if not os.path.exists(config_file):
                print(f"Config file not found at {config_file}")
                return None
            print(f"Using config file at {config_file}")
            # Try setting the environment variable and authenticate again
            os.environ['KAGGLE_CONFIG_DIR'] = config_path
            api.authenticate()
        
        return api
    except Exception as e:
        print(f"Error initializing Kaggle API: {e}")
        return None

# Global flag to track if we've downloaded the dataset in this session
DATASET_DOWNLOADED = False

def download_kaggle_dataset():
    """Download dataset from Kaggle if not already downloaded"""
    global DATASET_DOWNLOADED
    
    # Check if we've already downloaded the dataset in this session
    if DATASET_DOWNLOADED:
        print("Dataset already downloaded in this session")
        return True
    
    # Check if all required CSV files already exist
    all_files_exist = all(os.path.exists(file_info['filename']) for file_info in CSV_FILES_INFO)
    if all_files_exist:
        print("All required CSV files already exist, skipping download")
        DATASET_DOWNLOADED = True
        return True
        
    try:
        # Get authenticated API instance
        api = get_kaggle_api()
        if api is None:
            raise Exception("Failed to initialize Kaggle API")
        
        # Download the dataset
        print(f"Downloading dataset {KAGGLE_DATASET}...")
        
        # Use the dataset_download_files method without additional parameters
        api.dataset_download_files(
            dataset=KAGGLE_DATASET,
            path='.',
            unzip=True,
            quiet=False
        )
        
        # Verify the files were downloaded
        all_files_exist = all(os.path.exists(file_info['filename']) for file_info in CSV_FILES_INFO)
        if not all_files_exist:
            raise Exception("Some required files are missing after download")
        
        print("Dataset downloaded and verified successfully")
        DATASET_DOWNLOADED = True
        return True
    except Exception as e:
        error_msg = f"Error downloading Kaggle dataset: {str(e)}"
        print(error_msg)
        if hasattr(st, 'error'):
            st.error(error_msg)
        return False

def create_and_populate_db():
    """Creates and populates the SQLite database."""
    # Set up Kaggle credentials if we're in Streamlit Cloud
    if hasattr(st, 'secrets'):
        if not setup_kaggle_credentials():
            raise Exception("Failed to set up Kaggle credentials")
        
        # Download dataset from Kaggle
        if not download_kaggle_dataset():
            raise Exception("Failed to download Kaggle dataset")
    
    # Check if database exists and has tables
    db_exists = os.path.exists(DATABASE_NAME)
    
    conn = None
    try:
        # Create a connection to the SQLite database
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        if not db_exists:
            print(f"Database '{DATABASE_NAME}' created successfully.")
        else:
            print(f"Using existing database: {DATABASE_NAME}")

        # Load data from each CSV into a table
        for file_info in CSV_FILES_INFO:
            filename = file_info['filename']
            table_name = file_info['table_name']

            if not os.path.exists(filename):
                raise FileNotFoundError(f"CSV file not found: {filename}")

            # Check if table already exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                print(f"Table '{table_name}' already exists. Skipping...")
                continue

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
        return True

    except sqlite3.Error as e:
        error_msg = f"SQLite error: {e}"
        print(error_msg)
        if hasattr(st, 'error'):
            st.error(error_msg)
        return False
    except pd.errors.EmptyDataError:
        error_msg = "Error: One of the CSV files is empty or not found at the specified path."
        print(error_msg)
        if hasattr(st, 'error'):
            st.error(error_msg)
        return False
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(error_msg)
        if hasattr(st, 'error'):
            st.error(error_msg)
        return False
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

def load_data():
    """Loads data from the database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        print(f"Successfully connected to database: {DATABASE_NAME}")

        # --- Load songs data ---
        query_songs = "SELECT track_id, name, artist, tags, year, danceability, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo FROM songs_data"
        df_songs = pd.read_sql_query(query_songs, conn)
        print(f"Loaded 'songs_data' into DataFrame with {len(df_songs)} rows.")

        # --- Load interactions data ---
        query_interactions = "SELECT user_id, track_id, playcount FROM interactions_data"
        df_inter = pd.read_sql_query(query_interactions, conn)
        print(f"Loaded 'interactions_data' into DataFrame with {len(df_inter)} rows.")

        # --- Data Preprocessing ---

        # For df_songs:
        if 'track_idx' not in df_songs.columns:
            df_songs['track_idx'] = df_songs.index
        track2idx = pd.Series(df_songs.track_idx.values, index=df_songs[SCHEMA["track_id"]]).to_dict()
        idx2track = {v: k for k, v in track2idx.items()}

        # For df_inter:
        df_inter['track_idx'] = df_inter[SCHEMA["track_id"]].map(track2idx)

        if 'user_idx' not in df_inter.columns:
            user_ids = df_inter[SCHEMA["user_id"]].unique()
            user2idx = {user_id: i for i, user_id in enumerate(user_ids)}
            df_inter['user_idx'] = df_inter[SCHEMA["user_id"]].map(user2idx)

        df_inter.dropna(subset=['track_idx', 'user_idx'], inplace=True)
        df_inter['user_idx'] = df_inter['user_idx'].astype(int)
        df_inter['track_idx'] = df_inter['track_idx'].astype(int)

        print("Data loading and preprocessing complete.")
        return df_songs, df_inter, SCHEMA, track2idx, idx2track

    except sqlite3.Error as e:
        st.error(f"SQLite error during data loading: {e}")
        print(f"SQLite error during data loading: {e}")
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

# 🧠 Content-based filtering: Process each feature separately
def get_content_matrix(df_songs):
    try:
        # Get column names from SCHEMA
        artist_col = SCHEMA["artist"]
        tags_col = SCHEMA["tags"]
        year_col = SCHEMA["year"]
        feature_cols = SCHEMA["features"]
        
        # 1. Process artist column (categorical) - OneHotEncoder
        print(f"Processing artist column: {artist_col}")
        artist_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        artist_features = artist_encoder.fit_transform(df_songs[[artist_col]])
        print(f"Artist features shape: {artist_features.shape}")
        
        # 2. Process tags column (text) - TfidfVectorizer
        print(f"Processing tags column: {tags_col}")
        tags_vectorizer = TfidfVectorizer(max_features=1000)
        tags_features = tags_vectorizer.fit_transform(df_songs[tags_col].fillna(""))
        print(f"Tags features shape: {tags_features.shape}")
        
        # 3. Process year column (categorical) - OneHotEncoder
        print(f"Processing year column: {year_col}")
        year_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        year_features = year_encoder.fit_transform(df_songs[[year_col]])
        print(f"Year features shape: {year_features.shape}")
        
        # 4. Process audio features (numerical) - StandardScaler
        print(f"Processing audio feature columns: {feature_cols}")
        numeric_features = df_songs[feature_cols].fillna(0).values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features)
        # Convert to sparse matrix to be consistent with other features
        sparse_numeric = csr_matrix(scaled_features)
        print(f"Audio features shape: {sparse_numeric.shape}")
        
        # Combine all features using horizontal stack
        X_content = hstack([artist_features, tags_features, year_features, sparse_numeric])
        print(f"Combined content matrix shape: {X_content.shape}")
        
        return X_content
        
    except Exception as e:
        print(f"Error in get_content_matrix: {e}")
        st.error(f"Error creating content matrix: {e}")
        raise e

# 🧩 Collaborative filtering matrix
def build_collaborative_matrix(df_inter, n_items, n_users):
    try:
        print(f"Building collaborative matrix with {n_items} items and {n_users} users")
        
        # Ensure data types are correct
        plays = df_inter[SCHEMA["playcount"]].astype(float)
        track_indices = df_inter.track_idx.astype(int)
        user_indices = df_inter.user_idx.astype(int)
        
        # Check for invalid indices and filter them out
        if track_indices.max() >= n_items or user_indices.max() >= n_users:
            print(f"Warning: Some indices are out of bounds. Max track_idx: {track_indices.max()}, max user_idx: {user_indices.max()}")
            # Filter out rows with out-of-bounds indices
            valid_mask = (track_indices < n_items) & (user_indices < n_users)
            plays = plays[valid_mask]
            track_indices = track_indices[valid_mask]
            user_indices = user_indices[valid_mask]
            print(f"Filtered to {len(plays)} valid interactions")
        
        # Create the sparse matrix of user-item interactions
        R = csr_matrix(
            (plays, (track_indices, user_indices)),
            shape=(n_items, n_users)
        )
        print(f"User-item matrix shape: {R.shape}")
        
        # Calculate item-item similarity using cosine similarity
        print("Calculating item-item similarity matrix...")
        S_collab = cosine_similarity(R, dense_output=False)
        print(f"Collaborative similarity matrix shape: {S_collab.shape}")
        
        return S_collab
    except Exception as e:
        print(f"Error in build_collaborative_matrix: {e}")
        st.error(f"Error building collaborative matrix: {e}")
        raise e

# 🎯 Hybrid Recommender
def hybrid_recommend(seed_track_id, X_content, S_collab, α=0.7, k=10):
    try:
        if seed_track_id not in track2idx:
            print(f"Track ID {seed_track_id} not in track2idx")
            return []

        i = track2idx[seed_track_id]
        print(f"Generating recommendations for track idx {i}")
        
        # Get collaborative filtering scores
        collab_scores = S_collab[i].toarray().ravel()
        print(f"Collaborative scores shape: {collab_scores.shape}")
        
        # Get content-based scores
        content_scores = cosine_similarity(X_content[i:i+1], X_content).ravel()
        print(f"Content scores shape: {content_scores.shape}")
        
        # Ensure the arrays have the same length
        if len(collab_scores) != len(content_scores):
            min_len = min(len(collab_scores), len(content_scores))
            collab_scores = collab_scores[:min_len]
            content_scores = content_scores[:min_len]
            print(f"Adjusted score arrays to length {min_len}")
        
        # Combine scores with weighting
        hybrid = α * collab_scores + (1 - α) * content_scores
        
        # Get top recommendations
        top_idx = np.argsort(hybrid)[::-1]
        top_idx = [x for x in top_idx if x != i and x in idx2track][:k]
        
        return [idx2track[j] for j in top_idx]
    except Exception as e:
        print(f"Error in hybrid_recommend: {e}")
        return []

# 🚀 Run app
st.title("🎵 Spotify Hybrid Recommender")
st.markdown("This system recommends songs by combining collaborative filtering (user listening patterns) and content-based filtering (song features).")

# Create database and load data
if not create_and_populate_db():
    st.error("Failed to create or populate database. App cannot continue.")
    st.stop()

df_songs, df_inter, SCHEMA, track2idx, idx2track = load_data()
if df_songs is None or df_inter is None or SCHEMA is None:
    st.error("Failed to load data from database. App cannot continue.")
    st.stop()

# Precompute similarity matrices
try:
    with st.spinner("Building content-based similarity matrix..."):
        X_content = get_content_matrix(df_songs)
    
    with st.spinner("Building collaborative filtering matrix..."):
        n_items = df_songs.track_idx.max() + 1
        n_users = df_inter.user_idx.max() + 1
        print(f"Matrix dimensions: {n_items} items, {n_users} users")
        S_collab = build_collaborative_matrix(df_inter, n_items, n_users)
    
    st.success("Recommendation engine initialized successfully!")
except Exception as e:
    st.error(f"Error during precomputation: {e}")
    print(f"Detailed error: {e}")
    st.stop()

# UI Components
st.subheader("Find Songs You'll Love")

# Format function to show artist and title in the dropdown
def format_track_option(track_id):
    try:
        row = df_songs[df_songs[SCHEMA["track_id"]] == track_id].iloc[0]
        return f"{row[SCHEMA['track_name']]} - {row[SCHEMA['artist']]}"
    except:
        return track_id

# Create the UI elements
col1, col2 = st.columns([3, 1])
with col1:
    seed_track = st.selectbox(
        "Choose a seed track:", 
        options=df_songs[SCHEMA["track_id"]].unique(),
        format_func=format_track_option
    )

with col2:
    k = st.slider("Number of recommendations:", 5, 20, 10)
    
alpha = st.slider(
    "Balance between collaborative and content-based filtering:", 
    0.0, 1.0, 0.7, 0.05,
    help="1.0 = purely collaborative, 0.0 = purely content-based"
)

# Generate and display recommendations
if seed_track:
    # Get seed track info
    seed_info = df_songs[df_songs[SCHEMA["track_id"]] == seed_track].iloc[0]
    st.write(f"### Recommendations based on: {seed_info[SCHEMA['track_name']]} by {seed_info[SCHEMA['artist']]}")
    
    with st.spinner("Generating personalized recommendations..."):
        recs = hybrid_recommend(seed_track, X_content, S_collab, α=alpha, k=k)
    
    if not recs:
        st.warning("Could not generate recommendations for this track. Please try another.")
    else:
        # Display recommendations in a nice format
        st.subheader("Recommended Tracks:")
        
        # Create two columns for the recommendations
        cols = st.columns(2)
        
        for i, track_id in enumerate(recs):
            col_idx = i % 2  # Alternate between columns
            
            with cols[col_idx]:
                try:
                    # Get track info
                    row = df_songs[df_songs[SCHEMA["track_id"]] == track_id].iloc[0]
                    track_name = row[SCHEMA["track_name"]]
                    artist = row[SCHEMA["artist"]]
                    
                    # Display track with more details
                    with st.container():
                        st.markdown(f"**{i+1}. {track_name}** by *{artist}*")
                        
                        # Show song features in an expander
                        with st.expander("Show song details"):
                            # Format the audio features in a more readable way
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Year:** {row[SCHEMA['year']]}")
                                st.write(f"**Danceability:** {row['danceability']:.2f}")
                                st.write(f"**Energy:** {abs(row['loudness'])/10:.2f}")
                                st.write(f"**Speechiness:** {row['speechiness']:.2f}")
                            with col2:
                                st.write(f"**Acousticness:** {row['acousticness']:.2f}")
                                st.write(f"**Instrumentalness:** {row['instrumentalness']:.2f}")
                                st.write(f"**Valence:** {row['valence']:.2f}")
                                st.write(f"**Tempo:** {row['tempo']:.0f} BPM")
                except Exception as e:
                    print(f"Error displaying track {track_id}: {e}")
                    st.error(f"Error displaying track information")
        
        # Explain the recommendation balance
        st.info(f"Recommendations are weighted {int(alpha*100)}% from user listening patterns and {int((1-alpha)*100)}% from song attributes.")

