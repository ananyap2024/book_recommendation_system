# import libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import random
import math
import os

import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer


# page configuration
st.set_page_config(
    page_title="brs-streamlit",
    page_icon="🤖",
    layout="wide",
)

st.title("Book Recommender System")
st.caption("Popularity | Collaborative Filtering (Cosine/KNN/SVD) | Content based (TF/IDF)")


############################################
#"""## Helpers"""

# importing datasets
@st.cache_data(show_spinner=False)

def read_csv_safely(file_path):
  if file_path is None:
    return None
  return pd.read_csv(file_path)


# unique values
def unique_values(df):
  return pd.DataFrame({
      'Column' : df.columns,
      'Unique Values' : [df[c].nunique(dropna=True) for c in df.columns]
  })


# missing values
def missing_values(df):
  val = df.isnull().sum()
  per = df.isnull().mean()*100
  return pd.concat([val, per], axis=1, keys=["missing_values", "missing_percentage"])


# replacing values
def replace_values(df, index, col1, col2, col3, col4):
  temp = df.loc[index, col1]
  df.loc[index, col1] = df.loc[index, col2]
  try:
    df.loc[index, col2] = str(df.loc[index, col3]).split(';')[1]
  except Exception:
    pass
  df.loc[index, col4] = temp


# merging datasets; creating a merged dataframe
@st.cache_data(show_spinner=False)

def build_merged_df(df_books, df_ratings, df_users):
  # merge on primary colummns
  merged_df = pd.merge(df_users, df_ratings, on='User-ID')
  merged_df = pd.merge(merged_df, df_books, on='ISBN')

  # drop image url column
  for col in ['Image-URL-S', 'Image-URL-M', 'Image-URL-L']:
    if col in merged_df.columns:
      merged_df.drop(col, axis=1, inplace=True)

  # standardize column names
  merged_df.columns = merged_df.columns.str.strip().str.lower().str.replace("-", "_")

  # create a column country from location & drop location & replacing values in country
  if "location" in merged_df.columns:
    merged_df['country'] = merged_df['location'].str.split(', ').str[-1]

    merged_df.drop("location", axis=1, inplace=True)

    merged_df['country'] = (
        merged_df['country'].replace('', 'other').replace("n/a", 'other')
    )


  # missing author and publisher
  if 'publisher' in merged_df.columns:
    merged_df['publisher'] = merged_df['publisher'].fillna('Unkown')
  if 'book_author' in merged_df.columns:
    merged_df['book_author'] = merged_df['book_author'].fillna('Unkown')


  # year_of_publication datatype values
  indices = merged_df.index[merged_df['year_of_publication'].isin(["DK Publishing Inc","Gallimard"])]
  print('Location of string values in Year-Of-Publication\n',indices)
  for index in indices:
    replace_values(merged_df, index, 'year_of_publication', 'publisher', 'book_title', 'book_author')


  # handling outliers
  if 'age' in merged_df.columns:
    merged_df['age'] = pd.to_numeric(merged_df['age'])
    merged_df.loc[(merged_df['age'] < 10) | (merged_df['age'] > 90)] = np.nan

    NullAges = int(merged_df['age'].isnull().sum())
    if NullAges > 0:
      med = merged_df['age'].median()
      std_dev = merged_df['age'].std()
      random_ages = np.random.randint(
          low=int(med - std_dev),
          high=int(med + std_dev) + 1,
          size=NullAges
      )
      age_series = merged_df['age'].copy()
      age_series[age_series.isnull()] = random_ages
      merged_df['age'] = age_series
      print("Generated random ages")


  return merged_df

##################################################
#"""## Recommender Helpers"""

# popularity recommender helper
@st.cache_data(show_spinner = False)
def build_popularity_df(merged_df, min_ratings=200):
  df = merged_df.copy()
  df['avg_ratings'] = df.groupby("book_title")['book_rating'].transform("mean")
  df['no_of_ratings'] = df.groupby("book_title")['book_rating'].transform("count")
  popular_df = df[['book_title', 'avg_ratings', 'no_of_ratings']].drop_duplicates('book_title')
  popular_df = popular_df[popular_df['no_of_ratings'] > int(min_ratings)].sort_values('avg_ratings', ascending = False)
  return popular_df

# cosine similarity helper
@st.cache_data(show_spinner=False)
def build_cf_cosine(filtered_df, min_user_ratings=200, min_book_rating=50):
  # filter active users
  x = filtered_df.groupby('user_id').count()['book_rating'] > int(min_user_ratings)
  df = filtered_df[filtered_df['user_id'].isin(x[x].index) ]

  # filter popular books
  y = df.groupby('book_title').count()['book_rating'] >= int(min_book_rating)
  df = df[df['book_title'].isin(y[y].index)]

  pt = df.pivot_table(index='book_title', columns='user_id', values='book_rating').fillna(0)
  sim = cosine_similarity(pt)
  return df, pt, sim

def recommend_book_cosine(pt, similarity_scores, book_name, topn = 5):
  if book_name not in pt.index:
    return None
  idx = int(np.where(pt.index == book_name)[0][0])
  similar = sorted(list(enumerate(similarity_scores[idx])), key=lambda x:x[1], reverse=True)[1:topn+1]
  recs = [(pt.index[i[0]], float(i[1])) for i in similar]
  return pd.DataFrame(recs, columns=['Books', 'Similarity'])

# KNN model helper
@st.cache_data(show_spinner=False)
def build_knn_model(pt, n_neighbors = 11):
  knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=int(n_neighbors))
  knn.fit(pt)
  return knn

def recommend_book_knn(pt, knn, book_name, n_values=11):
  if book_name not in pt.index:
    return None
  distances, indices = knn.kneighbors(pt.loc[book_name,:].values.reshape(1, -1), n_neighbors=int(n_values))
  dist = distances.flatten().tolist()
  idx = indices.flatten().tolist()

  titles = []
  dists = []
  for i in range(1, len(idx)):
    titles.append(pt.index[idx[i]])
    dists.append(float(dist[i]))
  return pd.DataFrame({"Book": titles, "Distance": dists})

# SVD model helper
@st.cache_resource(show_spinner=False)
def build_svd_model(merged_df, min_interactions=50, k_factors=15, test_size = 0.2,random_state = 42):
  users_interactions_count_df = merged_df.groupby(['book_title', 'user_id']).size().groupby('user_id').size()
  users_with_enough_df = users_interactions_count_df[users_interactions_count_df >= int(min_interactions)].reset_index()[['user_id']]
  interactions_selected = merged_df.merge(users_with_enough_df, how='right', on='user_id')

  def smooth_user_preference(x):
    return math.log(1+x,2)

  interactions_full = interactions_selected.groupby(['book_title', 'user_id'])['book_rating'].sum().apply(smooth_user_preference).reset_index()
  if len(interactions_full) < 20:
    return None
  le = preprocessing.LabelEncoder()
  le.fit(merged_df['book_title'].unique())

  train, test = train_test_split(interactions_full, test_size=float(test_size), stratify=interactions_full['user_id'], random_state=int(random_state))

  train_df = train.copy()
  test_df = test.copy()
  train_df['book_title'] = le.transform(train_df['book_title'])
  test_df['book_title'] = le.transform(test_df['book_title'])

  users_items_pivot_df = train_df.pivot_table(index='user_id', columns='book_title', values='book_rating').fillna(0)
  users_items = users_items_pivot_df.values
  user_ids = list(users_items_pivot_df.index)

  k = int(min(k_factors, min(users_items.shape)-1))
  if k<2:
    return None

  U, sigma, Vt = svds(users_items, k=k)
  sigma = np.diag(sigma)
  all_user_predict = np.dot(np.dot(U, sigma), Vt)
  preds_df = pd.DataFrame(all_user_predict, columns=users_items_pivot_df.columns, index=user_ids).transpose()

  return {'le':le, 'preds_df': preds_df}

def recommend_svd_for_user(svd_obj, merged_df, user_ids, topn=10):
  le = svd_obj['le']
  preds_df = svd_obj['preds_df']

  if user_ids not in preds_df.index:
    return None

  already = merged_df.loc[merged_df['user_id'] == user_ids, 'book_title'].dropna().astype(str).unique().tolist()
  user_preds = preds_df.loc[user_ids].sort_values(ascending=False).reset_index()
  user_preds.columns = ['book_title_id', 'recStrength']
  user_preds['book_title'] = le.inverse_transform(user_preds['book_title_id'].astype(int))

  recs = user_preds[~user_preds['book_title'].isin(already)].head(topn)
  return recs[['book_title', 'recStrength']]

if False:
  # TF-IDF content helper
  @st.cache_data(show_spinner=False)
  def build_tfidf_content_model(filtered_df, min_ratings=200):
    df = filtered_df.copy()
    if 'no_of_ratings' not in df.columns:
      df['no_of_ratings'] = df.groupby('book_title')['book_rating'].transform('count')

    df = df[df['no_of_ratings'] > int(min_ratings)].reset_index(drop=True)
    if len(df) < 10:
      return None

    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
    corpus = (df['book_title'].fillna("")).astype(str) + " " + df['book_author'].fillna("").astype(str)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix.astype(np.float32), tfidf_matrix.astype(np.float32))
    return {'df':df, "cosine_similarity": cosine_sim}


  def recommend_tfidf(model_obj, book_name, topn=5):
    df = model_obj['df']
    cosine_sim = model_obj["cosine_sim"]

    if book_name not in df['book_title'].values:
      return None

    idx = int(df.index[df['book_title'] == book_name].tolist()[0])
    sim_order = cosine_sim[idx].argsort()[::-1]

    recs =[]
    for i in sim_order:
      title = df.loc[i, 'book_title']
      if title != book_name and title not in recs:
        recs.append(title)
        if len(recs) == topn:
          break

    return pd.DataFrame({"Book": recs})


#####################################################

#"""## Sidebar - Data input"""

st.sidebar.header("Data Input")
st.sidebar.write("Upload Books.csv, Users.csv, Ratings.csv to proceed")

books_file = st.sidebar.file_uploader("Upload Books.csv", type=['csv'])
ratings_file = st.sidebar.file_uploader("Upload Ratings.csv", type=['csv'])
users_file = st.sidebar.file_uploader("Upload Users.csv", type=['csv'])

st.sidebar.divider()
use_local = st.sidebar.checkbox("Use Local CSVs fromn a folder (if not uploading)", value=False)
local_path = st.sidebar.text_input("Local folder path (optional)", value="")

def get_local_or_uploaded(uploaded, filename):
  if uploaded is not None:
    return uploaded
  if use_local:
    folder = local_path.strip() if local_path.strip() else "."
    p = f"{folder}/{filename}"
    if os.path.exists(p):
      return p
  return None

books_src = get_local_or_uploaded(books_file, 'Books.csv')
ratings_src = get_local_or_uploaded(ratings_file, 'Ratings.csv')
users_src = get_local_or_uploaded(users_file, 'Users.csv')

if books_src is None or ratings_src is None or users_src is None:
  st.info("Please upload **Books.csv**, **Ratings.csv**, and **Users.csv** from the sidebar (or enable local loading).")
  st.stop()

df_books = read_csv_safely(books_src)
df_ratings = read_csv_safely(ratings_src)
df_users = read_csv_safely(users_src)

with st.spinner("Preparing merged dataset...."):
  merged_df = build_merged_df(df_books, df_ratings, df_users)

# Ensure required columns exist
required_cols = ['book_title', 'book_rating', 'user_id']
missing = [c for c in required_cols if c not in merged_df.columns]
if missing:
  st.error(f"Missing required columns after merge/cleanup: {(missing)}. Please check CSV column names")
  st.stop()


#"""# Tabs"""

tab1, tab2, tab3, tab4 = st.tabs(['Overview', 'EDA', 'Popularity', 'Recommend'])

with tab1:
  st.header("Dataset Overview")
  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Books rows", f"{len(df_books):,}")
  c2.metric("Ratings rows", f"{len(df_ratings):,}")
  c3.metric("Users rows", f"{len(df_users):,}")
  c4.metric("Merged rows", f"{len(merged_df):,}")

  with st.expander("Show data samples"):
    st.write("Books (top 5)")
    st.dataframe(df_books.head())
    st.write("Ratings (top 5)")
    st.dataframe(df_ratings.head())
    st.write("Users (top 5)")
    st.dataframe(df_users.head())
    st.write("Merged (top 5)")
    st.dataframe(merged_df.head())

  with st.expander("Duplicates and missing values"):
    st.write("Duplicate")
    st.write({
        "Books duplicates" : int(df_books.duplicated().sum()),
        "Ratings duplicates" : int(df_ratings.duplicated().sum()),
        "Users duplicates" : int(df_users.duplicated().sum())
    })
    st.write("Missing values (merged)")
    st.write(missing_values(merged_df))

  with st.expander("Unique values"):
    st.dataframe(unique_values(merged_df))


with tab2:
  st.header("Exploratory Data Analysis (EDA)")

  colA, colB = st.columns(2)

  with colA:
    if "age" in merged_df.columns:
      st.subheader("**Age distribution**")
      u = merged_df['age'].value_counts().sort_index()
      fig = plt.figure()
      plt.bar(u.index, u.values)
      plt.xlabel("Age")
      plt.ylabel("Count of Users")
      plt.xlim(xmin=0)
      st.pyplot(fig, clear_figure=True)
    else:
      st.info("No age column found")

  with colB:
    if "year_of_publication" in merged_df.columns:
      st.subheader("**Year of publication distribution**")
      fig = plt.figure()
      merged_df['year_of_publication'] = pd.to_numeric(merged_df['year_of_publication'], errors='coerce')
      y = merged_df.loc[merged_df['year_of_publication'] > 1800]['year_of_publication']

      if len(y) > 0:
        sns.histplot(y, bins=50, kde=True)
        plt.xlabel("Year of Publication")
        plt.ylabel("Count of Books")
        st.pyplot(fig, clear_figure=True)

      else:
        st.info("year_of_publication column not found")

  colC, colD = st.columns(2)

  with colC:
    st.subheader("**Ratings distribution**")
    fig = plt.figure()
    rating_counts = merged_df[merged_df['book_rating'] != 0]['book_rating'].value_counts().sort_index()
    sns.barplot(x=rating_counts.index, y=rating_counts.values)
    plt.xlabel('Rating')
    plt.ylabel('Number of ratings')
    st.pyplot(fig, clear_figure=True)

  with colD:
    if 'country' in merged_df.columns:
      st.subheader("**Top 5 countries**")
      vc = merged_df['country'].value_counts().head(5).reset_index()
      vc.columns = ['Country', 'Count']
      #st.dataframe(vc)
      fig, ax = plt.subplots()
      ax.pie(vc['Count'], labels=vc['Country'], autopct='%1.2f%%')
      ax.axis('equal')
      st.pyplot(fig, clear_figure=True)


with tab3:
  st.subheader("Popularity-based Recommender")
  min_r = st.slider("Minimum number of ratings per book (threshold)", 10, 1000, 200, step=10)
  popularity_df = build_popularity_df(merged_df, min_r)
  st.write(f"Books meeting the threshold: **{len(popularity_df)}**")
  topn = st.slider("How many books to show", 5, 100, 20, step=5)
  st.dataframe(popularity_df.head(int(topn)))


with tab4:
  st.subheader("Recommendation Engines")

  method = st.selectbox(
    "Choose a method",
    [
        "Collaborative (Cosine Similarity)",
        "Collaborative (KNN Item-to-Item)",
        "Collaborative (SVD Matrix Factorization)",
        "Content (TF-IDF Title + Author)"
    ]
  )

  filtered_df = merged_df.copy()

  if method == "Collaborative (Cosine Similarity)":
    st.markdown("### Cosine Similarity (Memory-based CF)")
    min_user = st.slider("Min ratings per user (active users)", 10, 500, 200, step=10)
    min_book = st.slider("Min ratings per book", 5, 200, 50, step=5)

    with st.spinner("Building pivot table and similarity matrix....."):
      df_cf, pt, sim = build_cf_cosine(filtered_df, min_user_ratings = min_user, min_book_rating=min_book)

      st.write(f"Pivot table shape: **{pt.shape[0]:,} books x {pt.shape[1]:,}users**")
      book_name = st.selectbox("Pick a book title", options=pt.index.tolist()[:5000])
      topn = st.slider("Top-N recommentions", 3, 20, 5)

      if st.button("Recommend (Cosine)", type='primary'):
        recs = recommend_book_cosine(pt, sim, book_name, topn=topn)
        if recs is not None:
          st.warning("Book not found in pivot table")
        else:
          st.dataframe(recs)

  elif method == "Collaborative (KNN Item-to-Item)":
    st.markdown("### KNN (Memory-based CF)")
    min_user = st.slider("Min ratings per user (active users)", 10, 500, 200, step=10)
    min_book = st.slider("Min ratings per book", 5, 200, 50, step=5)
    n_neighbors = st.slider("Neighbors to compute", 5, 30, 11)

    with st.spinner("Preparing pivot table....."):
      df_cf, pt, sim = build_cf_cosine(filtered_df, min_user_ratings = min_user, min_book_rating=min_book)

    with st.spinner("Training KNN model....."):
      knn = build_knn_model(pt, n_neighbors=n_neighbors)

    book_name = st.selectbox("Pick a book title", options=pt.index.tolist()[:5000])
    if st.button("Recommend (KNN)", type='primary'):
      recs = recommend_book_knn(pt, knn, book_name, n_values=n_neighbors)
      if recs is None:
        st.warning("Book not found in pivot table")
      else:
        st.dataframe(recs)

  elif method == "Collaborative (SVD Matrix Factorization)":
    st.markdown("### SVD Matrix Factorization (Model-based CF)")
    st.info("SVD can be heavier depending on dataset size. Use the button to run it on demand.")

    min_interactions = st.slider("Min interactions per user", 10, 200, 50, step=5)
    k_factors = st.slider("Number of latent factors(k)", 2, 50, 15, step=1)
    topn = st.slider("Top-N recommendations", 5, 50, 10, step=5)

    if st.button("Run SVD Training & Predict", type='primary'):
      with st.spinner("Training SVD model and building predictions matrix....."):
        svd_obj = build_svd_model(merged_df, min_interactions=min_interactions, k_factors=k_factors)
      if svd_obj is None:
        st.error("Not enough data after filtering to build SVD model. Reduce thresholds or check dataset.")
      else:
        user_list = sorted(list(set(svd_obj["preds_df"].columns.tolist())))
        user_id = st.selectbox("Select a user_id (from trained users)", options=user_list)

        if st.button("Get recommendations for this user"):
          recs = recommend_svd_for_user(svd_obj, merged_df, user_id, topn=topn)
          if recs is None:
            st.warning("User not found in predictions matrix")
          else:
            st.dataframe(recs)

  else:
    st.markdown("Content-based (TF-IDF) selected")
    st.info("Note: This method needs more storage.")

    if False:
      st.markdown("### Content-based Recommender (TF-IDF on Title + Author)")
      min_ratings = st.slider("Min ratings per book (keep popular books)", 10, 1000, 20, step=10)
      topn = st.slider("Top-N recommendations", 3, 20, 5)

      with st.spinner("Building TF-IDF model....."):
        model_obj = build_tfidf_content_model(filtered_df, min_ratings=min_ratings)

      if model_obj is None:
        st.error("Not enough data after filtering to build TF-IDF model. Reduce thresholds or check dataset.")
      else:
        titles = model_obj['df']['book_title'].dropna().astype(str).unique().tolist()
        book_name = st.selectbox("Pick a book title", options=titles[:5000])
        if st.button("Recommend (TF-IDF)", type='primary'):
          recs = recommend_tfidf(model_obj, book_name, topn=topn)
          if recs is None:
            st.warning("Book not found in TF-IDF matrix")
          else:
            st.dataframe(recs)


st.divider()
st.caption("Tip: If your dataset is very large, increase thresholds (active users / popular books) to speed up similarity computations.")