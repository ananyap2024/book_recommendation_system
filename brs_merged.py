# to run this project, open the kaggle notebook and run it there.
# kaggle datasets: https://www.kaggle.com/code/ananyapappu/brs-merged



"""# **Book Recommendation System**"""

# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse.linalg import svds
import random
from sklearn.feature_extraction.text import TfidfVectorizer

#####################################################################

# importing datasets
books = pd.read_csv(r"C:\Users\anany\OneDrive\Desktop\Ananya\projects\book_rec_sys\Books.csv", low_memory=False)
ratings = pd.read_csv(r"C:\Users\anany\OneDrive\Desktop\Ananya\projects\book_rec_sys\Ratings.csv")
users = pd.read_csv(r"C:\Users\anany\OneDrive\Desktop\Ananya\projects\book_rec_sys\Users.csv")

# merging users and ratings
merged_df = pd.merge(users, ratings, on='User-ID', how='inner')

# merging with books
merged_df = pd.merge(merged_df, books, on='ISBN', how='inner')

merged_df

# no. of rows & columns
print("\nShape",merged_df.shape)

# table details
print("Table Information\n")
merged_df.info()

# the first 5rows
print("First 5rows\n")
merged_df.head()

# checking for duplicates after merging
print("Checking Duplicates")
merged_df.duplicated().sum()

# checking for nulls in the dataset
print("Checking Nulls")
merged_df.isnull().sum()

#####################################################################

"""# **Data Cleaning**"""

# dropping url's, not needed for model training
merged_df = merged_df.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)

# removing spaces b/w words in book-titles
merged_df['Book-Title'] = merged_df['Book-Title'].str.replace(" ", "_", regex=False).str.strip().str.lower()
print("CLEANED BOOK TITLES\n",merged_df['Book-Title'])

# removing spaces b/w words in book-authors
merged_df['Book-Author'] = merged_df['Book-Author'].str.replace(" ", "_", regex=False).str.strip().str.lower()
print("\nCLEANED BOOK AUTHORS\n",merged_df['Book-Author'])

# removing spaces b/w words in publishers
merged_df['Publisher'] = merged_df['Publisher'].str.replace(" ", "_", regex=False).str.strip().str.lower()
print("\nCLEANED PUBLISHERS\n",merged_df['Publisher'])

# creating a new column country from location column
merged_df['Country'] = merged_df['Location'].str.split(',').str[-1].str.strip()

print("Country\n", merged_df['Country'])

# filling empty values of country
print("No. of nulls in Country", merged_df['Country'].isnull().sum())

merged_df['Country'] = (merged_df['Country'].fillna('other').str.replace('n/a', 'other', regex=False))

print("No. of empty values filled", (merged_df['Country'] == 'other').sum())
#apart from nan, they where n/a values in the column location

# dropping column location
merged_df = merged_df.drop(columns=['Location'])

# finding the missing values in merge_df
def missingvalues(df):
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df))*100

    missing_df = pd.DataFrame({"missing_count": missing_count, "missing_percentage": missing_percentage})

    return missing_df

print('Missing Values\n',missingvalues(merged_df))

# filling the book-author and pblisher values
merged_df['Book-Author'] = merged_df['Book-Author'].fillna("Unknown Author")

merged_df['Publisher'] = merged_df['Publisher'].fillna("Unknown Publisher")

print("Missing Values After Filling Author and Publisher\n", missingvalues(merged_df))

# handling outliers and missing values in AGE
# IQR method
Q1 = merged_df['Age'].quantile(0.25)
Q3 = merged_df['Age'].quantile(0.75)
IQR = Q3 - Q1

# outliers = (x < lb  |  x < ub)
outliers = merged_df[(merged_df['Age'] < Q1 - 1.5*IQR) | (merged_df['Age'] > Q3 + 1.5*IQR)]
print("Outliers of column - Age\n")
outliers

# checking outliers in column Age
plt.boxplot(merged_df['Age'].dropna(), vert=False)
#sns.boxplot(x=merged_df['Age'])
plt.title('Outliers of Age')
plt.show()

# setting a valid limit for age
merged_df.loc[(merged_df['Age']==0) | (merged_df['Age']<10) | (merged_df['Age']>90), 'Age'] = pd.NA

print("Min Age", merged_df['Age'].min())
print("Max Age", merged_df['Age'].max())

# checking outliers in column Age after setting limit
plt.boxplot(merged_df['Age'].dropna(), vert=False)
#sns.boxplot(x=merged_df['Age'])
plt.title('Boxplot of Age after setting limit')
plt.show()

# calculating median and standard-deviation for finding random integers
med = merged_df['Age'].median()
std_dev = merged_df['Age'].std()
print(med)
print(std_dev)

# generating random ages for empty values
NullAges = merged_df['Age'].isnull().sum()
print("No. of Null Ages",NullAges)

if NullAges > 0:
    random_ages = np.random.randint(
        low=int(med - std_dev),
        high=int(med + std_dev) + 1,
        size=NullAges
    )
    merged_df.loc[merged_df['Age'].isnull(), 'Age'] = random_ages
    print("Generated random ages")

else :
    print("No Null Ages to impute")

merged_df['Age'] = pd.to_numeric(merged_df['Age'], errors='coerce')

print("Missing values after cleaning data")
missingvalues(merged_df)

# finding the string dtype in year of publication  & replacing
indices = merged_df.index[merged_df['Year-Of-Publication'].isin(["DK Publishing Inc","Gallimard"])]
print('Location of string values in Year-Of-Publication\n',indices)

# Index([480243, 503165, 614114, 833991], dtype='int64')
#merged_df.iloc[[480243, 503165, 614114, 833991]]

# defining function to replace values for a given index
def replace_values(df, index, column):
    df.loc[index, column] = pd.NA

    return df

# replacing the string values of year-of-publication
replace_values(merged_df, indices, 'Year-Of-Publication')

# checking data of values changed
print('Number of values changed to null : ', merged_df['Year-Of-Publication'].isnull().sum())
print("\nThe rows in the data has been changed\n")
merged_df.loc[merged_df['Year-Of-Publication'].isna()]

merged_df['Year-Of-Publication'] = pd.to_numeric(merged_df['Year-Of-Publication'], errors='coerce')
#merged_df['Year-Of-Publication'].dtype
print("Max year:", merged_df['Year-Of-Publication'].max())
print("Min year:", merged_df['Year-Of-Publication'].min())


#####################################################################


"""# **EDA** (Exploratory Data Analysis)"""

print("Unique values of age : ", merged_df['Age'].nunique(), '\n')

# listing the frequency of each age
freq = merged_df['Age'].value_counts()
print(freq)

# sorting based on age - ascending
age_count = freq.sort_index()
print(age_count)

# plotting age vs freq
plt.figure(figsize=(16,6))
age_count.plot(kind='bar')
plt.title("Age vs Frequency")
plt.ylabel("Freq")
plt.show()

# setting a valid limit for year-of-publication
cur_date = datetime.now().year
merged_df.loc[(merged_df['Year-Of-Publication']==0) | (merged_df['Year-Of-Publication']<1000) | (merged_df['Year-Of-Publication']>cur_date), 'Year-Of-Publication'] = pd.NA

# listing and sorting the frequency of each year-of-publication
year_count = merged_df['Year-Of-Publication'].value_counts().sort_index()
print(year_count)

print('Latest Year',merged_df["Year-Of-Publication"].max())
print('Least Year', merged_df["Year-Of-Publication"].min())

# plotting freq vs year
plt.figure(figsize=(16,8))
year_count.plot(kind="bar")
plt.title("No. Of Books published in each year")
plt.show()

# plotting freq vs years in which freq>5
# more than 5 books published in one-year
freq_year_count = year_count[year_count > 100]
#print(freq_year_count)

plt.figure(figsize=(16,8))
freq_year_count.plot(kind="bar")
plt.title("No. of Books(>100) Printed in each year")
plt.show()

# top 10 authors
top_authors = merged_df['Book-Author'].value_counts().head(10).index

# plotting top 10 authors
plt.figure(figsize=(10, 6))
sns.countplot(
    data=merged_df,
    y='Book-Author',
    order=top_authors
)
plt.title("Top 10 Authors")
plt.xlabel("No. of Books")
plt.show()

# top 10 publishers
top_publishers = merged_df['Publisher'].value_counts().head(10).index

# plotting top 10 publishers
plt.figure(figsize=(10, 6))
sns.countplot(
    data=merged_df,
    y='Publisher',
    order=top_publishers
)
plt.title("Top 10 Publishers")
plt.xlabel("No. of Books")
plt.show()

# there were locations with no country name, so setting the country name as other.
merged_df.loc[merged_df['Country'] == '', 'Country'] = 'other'

# top 5 countries
top_countries = merged_df['Country'].value_counts().head(5)
print(top_countries)

# plotting top 5 countries
plt.pie(top_countries, labels = top_countries.index , autopct='%1.2f%%')
plt.title("Top 5 Countries with max users")
plt.axis()

# Analyzing famous books
book_stats = (
    merged_df
    .groupby(['Book-Title', 'Book-Author'])
    .agg(rating_count=('Book-Rating', 'count'),
         avg_rating=('Book-Rating', 'mean')
    ).reset_index()
)
print("Famous Books based on ratings")
book_stats

# setting limit for rating (atleast 50 users ratings)
min_ratings = 50
popular_books = book_stats[book_stats['rating_count'] >= min_ratings]

# top 10 books from popular books
top_books = popular_books.sort_values(by='rating_count',ascending=False).head(10)
print("Top 10 Popular Books with >50 user ratings")
top_books

# renaming title for book rating
top_books['Book'] = (top_books['Book-Title'] + " — " + top_books['Book-Author'])

# plotting top book
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_books,
    x='avg_rating',
    y='Book'
)
plt.xlabel('Average Rating')
plt.ylabel('Book')
plt.title('Top 10 Famous Books (High Ratings & Popularity)')
plt.show()

# calculating overall ratings of books
overall_rt = merged_df[merged_df['Book-Rating'] != 0]

# plotting overall rating
plt.figure(figsize=(8,4))
sns.countplot(data = overall_rt, x='Book-Rating')
plt.title("Overall Ratings of Books")
plt.ylabel("No. of ratings")
plt.show()


#####################################################################


"""# **Popularity-based recommender**"""

# average rating per book title
avg_rating = (merged_df.groupby('Book-Title')['Book-Rating'].mean().reset_index()).sort_values(by='Book-Rating', ascending=False)
print("Average ratings")
print(avg_rating)

# no of ratings per book title
no_of_ratings = (merged_df.groupby('Book-Title')['Book-Rating'].count().reset_index()).sort_values(by='Book-Rating', ascending=False)
print("\nNo of ratings")
print(no_of_ratings)

# popular books based of avg-ratings & no of ratings
popular_df = pd.DataFrame({'Book-Title':avg_rating['Book-Title'], 'Average-Ratings': avg_rating['Book-Rating'],'No. of Ratings':no_of_ratings['Book-Rating']})
popular_df = popular_df.sort_values(by='No. of Ratings', ascending=False)
print("New DF with avg_rating & no_of_rating")
popular_df

# checking for duplicates in popular_df
popular_df['Book-Title'].duplicated().sum()

# considering books with more that 200 ratings
popular_books = popular_df[popular_df['No. of Ratings'] >= 200]
print("DF with no_of-rating > 200")
popular_books

# sorting the books based on avg_ratings
popular_books = popular_books.sort_values(by='Average-Ratings', ascending=False)
print("sorted DF based on avg_ratings")
popular_books

# displaying top 10 recommended books based on average ratings
top = popular_books.head(10)

# avg rating vs book title
plt.figure(figsize=(10,6))
sns.barplot(
    data= top,
    x='Average-Ratings',
    y='Book-Title'
)
plt.title("Book vs Rating")
plt.show()

# displaying top 10 recommended books based on no. of ratings
top = popular_books.head(10)

# no. of ratings vs book title
plt.figure(figsize=(10,6))
sns.barplot(
    data= top,
    x='No. of Ratings',
    y='Book-Title'
)
plt.title("Book vs No. of Ratings")
plt.show()


#####################################################################


"""# **Collaborative Filtering(memory-based)**"""

#1 active users who gave more than 200 ratings
active_users = merged_df['User-ID'].value_counts().sort_values(ascending=True)
active_users = active_users[active_users>200]
print("active users who gave more than 200 ratings")
active_users

#2 books with atleast 50 users ratings
min_ratings = 50
popular_books = book_stats[book_stats['rating_count'] >= min_ratings]
print("books with atleast 50 users ratings")
popular_books

#3a creating a dataframe containg only the rows where user_rating>200 and book_rating>50
active_users = merged_df['User-ID'].value_counts()
active_users = active_users[active_users > 200].index

filtered_df = merged_df[merged_df['User-ID'].isin(active_users)]
#print("DF with ratings>200 ")
filtered_df

#3b creating a dataframe containg only the rows with books atleast 50 users ratings
popular_books = filtered_df['Book-Title'].value_counts()
popular_books = popular_books[popular_books > 50].index

filtered_df = filtered_df[filtered_df['Book-Title'].isin(popular_books)]
filtered_df

#3 user-item rating pivot-table
pt = filtered_df.pivot_table(index = 'Book-Title', columns = 'User-ID', values = 'Book-Rating', fill_value=0)
pt

#4 cosine_similarity matrix
similarity_scores = cosine_similarity(pt)
#similarity_scores

#5a creating df from similarity-scores array
similarity_df = pd.DataFrame(
    similarity_scores,
    index=pt.index,
    columns=pt.index
)
#similarity_df

#5b finding books based on cosine-similarities
def recommend_books(book_name, similarity_df=similarity_df, n =10):
    if book_name not in similarity_df.index:
        return f"Book '{book_name}' not found"

    similarity_score = similarity_df.loc[book_name]
    sorted_scores = similarity_score.sort_values(ascending=False)
    sorted_scores = sorted_scores.drop(book_name)

    recommended_books = sorted_scores.head(n).index.tolist()

    return recommended_books

#5 finding recommended books
book = "1984"
similar_books = recommend_books(book)
print(f"Books similar to '{book}'")
similar_books

#6 finding similarity scores
def find_similar_books(similar_books, similarity_df):
    #similarity_values = []
    for book in similar_books:
        similarity = similarity_df.loc[book, similar_books]
        #similarity_values.append(similarity)

    df = pd.DataFrame({'Book':similar_books, 'Similarity':similarity})
    df = df.reset_index(drop=True)

    return df

print("Books with Similar similary_scores")
find_similar_books(similar_books, similarity_df)

# test case2
#5 finding recommended books
book = "To Kill a Mockingbird"
similar_books = recommend_books(book)
print(f"Books similar to '{book}'")
similar_books


#####################################################################


"""# **KNN Recommender**"""

# knn item-to-item recommender
# 1. creating a knn model

knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
knn.fit(pt)

#2. creating a recommendation function
def recommend(book_title, n_values=10):
    if book_title not in pt.index:
        return f"Book {book_title} not found."

    book_vector = pt.loc[book_title].values.reshape(1, -1)

    distances, indices = knn.kneighbors(book_vector, n_neighbors = n_values)

    recommended_books = pt.index[indices.flatten()]
    similarity_scores = 1 - distances.flatten()

    result_df = pd.DataFrame({'Book-Title': recommended_books, 'Similarity': similarity_scores}).iloc[1:]

    return result_df

# printing similar books using KNN
book_title = '1984'
print(f"Books similar to '{book_title}'")
recommend(book_title)

# test case
recommend(book_title='1985')


#####################################################################


"""# **Model-based collaborative filtering (SVD/matrix factorization)**"""

# using filtered_df -> min 200 ratings each user has given
#filtered_df = df_active
df_active = filtered_df
df_active

# grouping ratings by book_title & user_id
interactions_full_df = (df_active.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean().reset_index())

# log smoothing - reduces impact of extreme ratings
interactions_full_df['Book-Rating'] = np.log2(1 + interactions_full_df['Book-Rating'])

print("Grouped_df based on book-title & user-id\n")
interactions_full_df

# used to get >=2 users for stratification
user_counts = interactions_full_df['User-ID'].value_counts()
valid_users = user_counts[user_counts >= 2].index

filtered_interactions = interactions_full_df[
    interactions_full_df['User-ID'].isin(valid_users)
]

# train–test split with stratification by user
train_df, test_df = train_test_split(filtered_interactions, test_size=0.2, random_state=42, stratify=filtered_interactions['User-ID'])
#train_df, test_df = train_test_split(interactions_full_df, test_size=0.2, random_state=42)

# label-encoding book_titles -> converting non-numerica data into numeric data
book_encoder = LabelEncoder()
book_encoder.fit(merged_df['Book-Title'])
train_df['book_title_id'] = book_encoder.transform(train_df['Book-Title'])
test_df['book_title_id'] = book_encoder.transform(test_df['Book-Title'])

# building sparse user–item matrix (training set)
user_item_matrix = train_df.pivot_table(index='User-ID', columns='book_title_id', values='Book-Rating', fill_value=0)
print("sparse user–item matrix\n")
user_item_matrix

# running SVD on the user–item matrix
NUMBER_OF_FACTORS_MF = 50

U, sigma, Vt = svds(user_item_matrix.values, k=NUMBER_OF_FACTORS_MF)

sigma = np.diag(sigma)
sigma

"""1. **NUMBER_OF_FACTORS_MF = 50**
   > Latent Factor Selection -> captures underlying user preferences and book characteristics
2. **U, sigma, Vt = svds(user_item_matrix.values, k=NUMBER_OF_FACTORS_MF)**
   > Truncated SVD decomposition -> svds(R, k=50) factorizes the sparse user–item matrix into user factors (U), singular values (σ), and item factors (Vᵀ)
3. **sigma = np.diag(sigma)**
   > Rating reconstruction for recommendations -> Converting σ into a diagonal matrix
"""

# Reconstruct predicted ratings
# Predictions = U × Σ × VT
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
all_user_predicted_ratings

"""**all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)**
> This line reconstructs the full predicted user–item rating matrix by multiplying user latent factors, factor strengths, and item latent factors obtained from truncated SVD.
"""

# Convert predictions back to a DataFrame
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)
cf_preds_df


#####################################################################


"""# **Wrapping SVD predictions into recommender class**
**Model-Based Filtering Recommender**
"""

class CFRecommender:
    def __init__(self, cf_preds_df, interactions_df):
        """
        cf_preds_df: DataFrame of predicted ratings (users × items)
        interactions_df: Original interaction data
        """
        self.cf_preds_df = cf_preds_df
        self.interactions_df = interactions_df

    def recommend_items(self, user_id, items_to_ignore=None, topn=10, verbose=False):
        if user_id not in self.cf_preds_df.index:
            raise ValueError(f"User {user_id} not found in prediction matrix.")

        # Predicted ratings for user
        user_preds = self.cf_preds_df.loc[user_id]

        # Sort predictions descending
        user_preds_sorted = user_preds.sort_values(ascending=False)

        # Remove already interacted items
        if items_to_ignore is not None:
            user_preds_sorted = user_preds_sorted.drop(
                labels=items_to_ignore,
                errors='ignore'
            )

        # Top-N recommendations
        top_recommendations = user_preds_sorted.head(topn)

        if not verbose:
            return list(top_recommendations.index)

        return pd.DataFrame({
            'Book-Title': top_recommendations.index,
            'Predicted-Rating': top_recommendations.values
        })

cf_recommender_model = CFRecommender(
    cf_preds_df=cf_preds_df,
    interactions_df=filtered_df
)

sample_user_id = 254

items_rated_by_user = (
    filtered_df[filtered_df['User-ID'] == sample_user_id]['Book-Title']
    .unique()
)

recommendations = cf_recommender_model.recommend_items(
    user_id=sample_user_id,
    items_to_ignore=items_rated_by_user,
    topn=5,
    verbose=True
)

recommendations


#####################################################################


"""# **Evaluate the SVD Recommender**"""

train_indexed_df = train_df.set_index('User-ID')
test_indexed_df = test_df.set_index('User-ID')

def get_items_interacted(user_id, interactions_df):
    try:
        return set(interactions_df.loc[user_id]['Book-Title'])
    except KeyError:
        return set()

class ModelEvaluator:
    def __init__(self, train_df, test_df, all_items):
        self.train_df = train_df
        self.test_df = test_df
        self.all_items = all_items

    def get_not_interacted_items_sample(self, user_id, sample_size, seed=42):
        interacted_items = get_items_interacted(user_id, self.train_df)
        non_interacted = list(self.all_items - interacted_items)

        random.seed(seed)
        return set(random.sample(non_interacted, min(sample_size, len(non_interacted))))

    def evaluate_test_item(self, recommender, user_id, test_item, k):
        non_interacted_items = self.get_not_interacted_items_sample(user_id, 100)
        items_to_rank = non_interacted_items | {test_item}

        recommendations = recommender.recommend_items(
            user_id=user_id,
            items_to_ignore=None,
            topn=k,
            verbose=False
        )

        return int(test_item in recommendations)

    def evaluate_model(self, recommender, k_values=[5, 10]):
        hits = {k: [] for k in k_values}

        for user_id in self.test_df.index.unique():
            test_items = get_items_interacted(user_id, self.test_df)

            for test_item in test_items:
                for k in k_values:
                    hit = self.evaluate_test_item(recommender, user_id, test_item, k)
                    hits[k].append(hit)

        for k in k_values:
            print(f"Recall@{k}: {np.mean(hits[k]):.4f}")

all_books = set(filtered_df['Book-Title'].unique())

evaluator = ModelEvaluator(
    train_df=train_indexed_df,
    test_df=test_indexed_df,
    all_items=all_books
)

evaluator.evaluate_model(cf_recommender_model)


#####################################################################


"""# **Content-Based Filtering**
**TF-IDF on title + author**
"""

book_rating_counts = merged_df['Book-Title'].value_counts()

popular_books = book_rating_counts[book_rating_counts >= 200].index

content_df = merged_df[
    merged_df['Book-Title'].isin(popular_books)
][['Book-Title', 'Book-Author', 'ISBN']].drop_duplicates()

content_df = content_df.reset_index(drop=True)

content_df['text'] = (
    content_df['Book-Title'] + ' ' + content_df['Book-Author']
)

tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    stop_words='english'
)

tfidf_matrix = tfidf.fit_transform(content_df['text'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def book_recommender(book_title, top_n=5):
    if book_title not in content_df['Book-Title'].values:
        return f"Book '{book_title}' not found."

    idx = content_df[
        content_df['Book-Title'] == book_title
    ].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores, key=lambda x: x[1], reverse=True
    )

    sim_scores = sim_scores[1:top_n + 1]

    book_indices = [i[0] for i in sim_scores]

    return content_df.iloc[book_indices][
        ['Book-Title', 'Book-Author']
    ]

book_recommender("1984")