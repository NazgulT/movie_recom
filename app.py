# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import urllib.request
from surprise import Dataset, Reader, SVD
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random

# ========================================
# 1. CACHED DATA & MODEL LOADING
# ========================================

@st.cache_data
def download_movielens():
    base_url = "https://files.grouplens.org/datasets/movielens/ml-100k/"
    files = {'u.data': 'u.data', 'u.item': 'u.item'}
    for fname, local in files.items():
        if not os.path.exists(local):
            st.write(f"Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, local)
    return True

@st.cache_data
def load_data():
    ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    movies = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None,
                         usecols=[0, 1], names=['item_id', 'title'])
    df = ratings.merge(movies, on='item_id')
    return df, movies

@st.cache_resource
def train_svd_model(_df):
    print("Training SVD model... (this runs once)")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(_df[['user_id', 'item_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    model.fit(trainset)
    return model

@st.cache_resource
def load_hf_models():
    print("Loading Hugging Face text generation model... (this runs once)")
    t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    t5_pipe = pipeline(
        "text2text-generation", model=t5_model, tokenizer=t5_tokenizer,
        max_length=60, temperature=0.8, do_sample=True,
        device=0 if torch.cuda.is_available() else -1
    )
    return t5_pipe

# ========================================
# 2. LOAD EVERYTHING
# ========================================
download_movielens()
df, movies = load_data()
model = train_svd_model(df)
t5_pipe = load_hf_models()

# ========================================
# 3. GENRE INFERENCE - edit to get real genres
# ========================================
def infer_genres(title):
    title = title.lower()
    if any(x in title for x in ['star wars', 'matrix', 'blade runner', 'alien']): 
        return ["sci-fi", "action"]
    if any(x in title for x in ['godfather', 'shawshank', 'schindler']): 
        return ["drama"]
    if any(x in title for x in ['toy story', 'lion king']): 
        return ["animation", "family"]
    return ["action", "drama"]

# ========================================
# 4. TAGLINE GENERATOR
# ========================================
def generate_tagline(movie_title, user_id):
    user_ratings = df[df['user_id'] == user_id]
    if user_ratings.empty or user_ratings['rating'].mean() < 3.5:
        genres = ["any"]
    else:
        sample = user_ratings.sort_values('rating', ascending=False).head(3)
        genres = []
        for title in sample.merge(movies)['title']:
            genres.extend(infer_genres(title))
        genres = list(set(genres))[:2]
    genre_str = " and ".join(genres) if len(genres) > 1 else genres[0]
    
    prompt = f"Write a short movie tagline for '{movie_title}' for a fan of {genre_str} films. 10-20 words."
    try:
        base = t5_pipe(prompt, num_return_sequences=1)[0]['generated_text']
    except:
        base = f"Discover {movie_title} – a cinematic masterpiece!"
    return base.capitalize()

# ========================================
# 5. RECOMMENDATION ENGINE
# ========================================
def get_top_n_recommendations(user_id, n=5):
    user_id = int(user_id)
    all_items = df['item_id'].unique()
    rated_items = df[df['user_id'] == user_id]['item_id'].unique().tolist()
    unrated_items = [item for item in all_items if item not in rated_items]
    
    if not unrated_items:
        st.warning("This user rated everything! Showing popular movies.")
        popular = df.groupby('title')['rating'].mean().sort_values(ascending=False).head(n)
        return popular.index.tolist(), popular.values.tolist()
    
    predictions = [(item, model.predict(user_id, item).est) for item in unrated_items]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    top_items = [p[0] for p in top_n]
    scores = [p[1] for p in top_n]
    
    recs = movies[movies['item_id'].isin(top_items)].copy()
    recs['predicted_rating'] = scores
    recs = recs.sort_values('predicted_rating', ascending=False)
    return recs['title'].tolist(), recs['predicted_rating'].tolist()

# ========================================
# 6. PLOTLY NETWORK GRAPH (REPLACES BOKEH)
# ========================================
def plot_user_graph_plotly(user_id):
    user_id = int(user_id)
    user_data = df[(df['user_id'] == user_id) & (df['rating'] >= 4.0)]
    if user_data.empty:
        return None
    
    # Build graph
    G = nx.from_pandas_edgelist(
        user_data, source='user_id', target='title', edge_attr='rating'
    )
    G.add_node(user_id)
    
    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
    
    # Node data
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        if node == user_id:
            node_text.append(f"<b>User {user_id}</b>")
            node_color.append("red")
            node_size.append(30)
        else:
            rating = G[user_id][node]['rating'] if node in G[user_id] else 0
            node_text.append(f"{node}<br>Rating: {rating}")
            node_color.append(rating)
            node_size.append(20)
    
    # Edge data
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create figure
    fig = go.Figure()
    
    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[f"<b>{t.split('<')[0]}</b>" if "User" in t else t.split('<')[0] for t in node_text],
        textposition="top center",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='Viridis',
            colorbar=dict(title="Rating"),
            line=dict(width=2, color='black')
        ),
        hovertemplate=node_text,
        hoverinfo="text"
    ))
    
    fig.update_layout(
        title=f"User {user_id}'s High-Rated Movies (≥4.0)",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    
    return fig

# ========================================
# 7. STREAMLIT UI
# ========================================
st.set_page_config(page_title="AI Movie Recommender", layout="wide")
st.title("AI-Powered Movie Recommender")
st.markdown("Enter a **User ID** (1–943) to get personalized recommendations with **AI-generated taglines** and **interactive taste graph**.")

# Sidebar
st.sidebar.header("User Input")
user_id = st.sidebar.number_input("Enter User ID (1–943)", 1, 943, 1)
n_recs = st.sidebar.slider("Number of Recommendations", 3, 10, 5)


if st.sidebar.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        try:
            titles, scores = get_top_n_recommendations(user_id, n_recs)
            
            st.success(f"Top {n_recs} Movies for User {user_id}")
            for title, score in zip(titles, scores):
                with st.expander(f"**{title}** (Predicted: {score:.2f}/5.0)"):
                    description = generate_tagline(title, int(user_id))
                    st.caption(f"*{description}*")
            
            
            st.subheader("Your Movie Taste Network")
            graph = plot_user_graph_plotly(user_id)
            if graph:
                st.plotly_chart(graph, width='stretch')
            else:
                st.info("No high ratings (≥4.0) to display graph.")
        
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure User ID is between 1 and 943.")
# ========================================
# 8. FOOTER
# ========================================
st.markdown("---")
st.caption("Built by Naz with **Surprise SVD**, **Hugging Face**, **Plotly**, and **Streamlit** | "
           "Dataset: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)")