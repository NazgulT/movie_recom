# AI-Powered Movie Recommender with Personalized Taglines

**Collaborative Filtering + Generative AI + Interactive Visualizations**

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"For the sci-fi fan who dreams in binary…"**  
> — AI-generated tagline for *Blade Runner* (User 42)

---

This **AI-powered movie recommender** uses **SVD collaborative filtering** on the MovieLens 100K dataset to predict personalized movie ratings. It delivers **top-N recommendations** with confidence scores and generates **creative taglines** using **Hugging Face FLAN-T5** offline, without an API key. An **interactive Plotly network graph** visualizes user taste. Built with **Streamlit**, the app **caches models and data** for instant reloads.

This project includes EDA on the movie dataset.

---

## Demo GIF

![Demo GIF](https://github.com/user-attachments/assets/ced90185-e41d-4b6b-830e-86203cc43c2d)

> Watch the app in action: Enter User ID = 5, get 5 personalized recs with AI taglines, and explore the taste graph!

---

## Features

- **Personalized Top-N Recommendations** using **Singular Value Decomposition (SVD)**
- **AI-Generated Taglines** via **FLAN-T5** (100% free, no API key)
- **Interactive Taste Network Graph** with **Plotly**
- **Fully Cached** – model trains once, loads instantly
- **EDA** on MovieLens 100K

---

## Project Structure

- app.py #Main Streamlit App
- requirements.txt #Project Dependencies
- movie-lens-EDA.ipynb #Movie-Lens EDA
- README.md

## How It Works

- SVD Model learns latent factors from ratings
- Predicts unseen movie scores
- Top-N filtered & ranked
- Hugging Face generates creative taglines
- Plotly draws your movie taste network

## App Screenshots

### Recommendations + AI Tagline

<img width="1416" height="566" alt="recom" src="https://github.com/user-attachments/assets/427d1a37-5386-4ea2-9423-20e7a5b348b0" />

### Interactive Taste Network

<img width="1408" height="688" alt="taste_network" src="https://github.com/user-attachments/assets/26b66b40-64d4-4f99-b201-541f6afc2f08" />

## Single Sample Output

<img width="1050" height="119" alt="tagline" src="https://github.com/user-attachments/assets/16520d82-3556-481d-a5d7-41c6315ba006" />


## Exploratory Data Analysis (EDA)

### 1. Dataset Overview
```text
Rows: 100,000 ratings
Users: 943
Movies: 1,682
Rating Scale: 1–5 stars
Rating Duration: 1997-1998
```

### 2. Top 10 Most Rated Movies

<img width="1189" height="590" alt="top10mov" src="https://github.com/user-attachments/assets/076cb2b6-b252-43a2-ab70-254b1affb763" />

_______
Insight: 
- Blockbusters dominate the ratings

### 3. Rating Distribution

<img width="482" height="502" alt="ratings_dist_pie" src="https://github.com/user-attachments/assets/e6ec7cef-c7a8-48cb-9352-8f90c0e47916" />

_________
Insight: 
- 4-star ratings are common - users tend to rate positively. Extreme ratings 1.0 and 5.0 are less common, which is typical in rating systems.

### 4. Movie Popularity

<img width="1005" height="545" alt="movie_popularity" src="https://github.com/user-attachments/assets/0df75850-eabc-4a7d-81db-8e1ed0467b54" />

________
Key Insight:
- The chart follows the Power Law distribution: Top movies get many ratings, while most movies get very few ratings.

### 5. User Activity

<img width="1005" height="545" alt="user_activity" src="https://github.com/user-attachments/assets/6b023cad-11c2-480b-94ad-27beecb08ea5" />

_________
Insight:
- The above histogram of user activity follows the Power Law Distribution with its specific long tail and tall head.
- This means that the most of the ratings were owned by the small amount of users. Specifically, few users rate many movies.
- The most active users and the number of ratings they gave:

  
| User ID    | Num of ratings |
| -------- | ------- |
| 405  |  737 |
| 655 | 685   |
| 13    | 636    |
| 450 | P540 |
| 276 | 518 |

### 6. Time-based analysis

<img width="1023" height="525" alt="ratings_hourwise" src="https://github.com/user-attachments/assets/37c16970-6162-467f-85d6-8319b5602ba1" />

___________
Insight: Users rated the most in the evening and at night. 


## Tech Stack

| Layer    | Tool |
| -------- | ------- |
| ML  | Surprise (SVD)    |
| GenAI | Hugging Face Transformers     |
| Frontend    | Streamlit    |
| Visualizations | Plotly, Matplotlib, Seaborn

## Quick Start

### 1. Clone
``` python
git clone https://github.com/your-username/movie-recommender.git`
cd movie-recom
```

### 2. Install
``` python
pip install -r requirements.txt
```

### 3. Run
``` python 
streamlit run app.py
```

First run lasts about 3 minutes. 
Subsequent runs will run instantly.

## Future Ideas

- Add movie posters (OMDb API)
- Use IMDb dataset, webscrape Netflix
- Use genre information more extensively
- Use GridSearch to find the best model, evaluate

Author

Nazgul Sagatova, Data Scientist

nazgul.tazhigaliyeva@gmail.com
