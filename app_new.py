import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from data_loader import load_data  # This is in a separate file named data_loader.py

# Set page title and configuration
st.set_page_config(page_title="Spotify Recommender System", layout="wide")

# 🎒 Load data using schema-driven loader
try:
    st.info("Loading data... Please wait.")
    df_songs, df_inter, SCHEMA, track2idx, idx2track = load_data()
    # Debug information
    print("Columns in df_songs:", df_songs.columns.tolist())
    print("SCHEMA keys:", SCHEMA)
    
    # Verify actual feature columns exist in the dataframe
    feature_cols = SCHEMA["features"]
    for col in feature_cols:
        if col not in df_songs.columns:
            st.error(f"Column '{col}' not found in song dataframe. Available columns: {df_songs.columns.tolist()}")
            st.stop()
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

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
