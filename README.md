# 📚 Book Recommendation System

📚 Project Overview 
It explores multiple recommendation strategies, starting from simple popularity-based methods to advanced collaborative filtering and content-based approaches.

🔑 Key Components
1. **Data Cleaning**
- Preprocessing raw book and user interaction data.
- Handling missing values, duplicates, and formatting inconsistencies.
- Ensuring the dataset is structured for recommendation tasks.

2. **Exploratory Data Analysis (EDA)**
- Understanding dataset distribution (books, ratings, users).
- Identifying trends such as most popular books, rating patterns, and user activity.
- Visualizing insights to guide model selection.

3. **Popularity-Based Recommender**
- Simple baseline model.
- Recommends books based on overall popularity (e.g., number of ratings or average rating).
- Useful for new users with no prior history (cold-start problem).

4. **Collaborative Filtering (Memory-Based)**
- Uses user-user or item-item similarity.
- Example: KNN (k-nearest neighbors) to find similar users/books.
- Relies on rating patterns to suggest books.

5. **Model-Based Collaborative Filtering**
- SVD (Singular Value Decomposition) / Matrix Factorization applied to rating matrices.
- Captures latent features (hidden patterns in user-book interactions).
- More scalable and effective than memory-based methods.

6. **SVD Recommender Class**
- Wrapping SVD predictions into a reusable recommender class.
- Provides structured methods for training, predicting, and generating recommendations.

7. **Evaluation**
- Measuring accuracy of recommendations using metrics like RMSE, precision, recall, or top-N accuracy.
- Comparing performance of popularity-based, memory-based, and model-based approaches.

8. **Content-Based Filtering**
- Uses book metadata (e.g., genre, author, description).
- Matches user preferences with similar book attributes.
- Helps overcome limitations of collaborative filtering when rating data is sparse.
- This project contains two implementations:

--- 

1. **Core Recommendation Logic** – implemented in `brs_merged.py`
2. **Interactive Web Application** – implemented using **Streamlit**

---

# 📂 Project Structure

```
book-recommendation-system
│
├── data
│   ├── Books.csv
│   ├── Ratings.csv
│   └── Users.csv
│
├── src
│   ├── brs_merged.py
│   └── brs_streamlit.py
│
├── README.md
├── STREAMLIT.md
└── requirements.txt
```

---

# 🧠 Recommendation Approach

The system uses the following steps:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Popularity-based recommender
- Collaborative filtering
- KNN recommender
- SVD recommender
- Content-based recommendation

Libraries used in the project:

- pandas
- numpy
- scikit-learn
- streamlit
- matplotlib
- seaborn

---

# 📊 Dataset

The dataset used in this project contains the following files:

| File        | Description                      |
|-------------|----------------------------------|
| Books.csv   | Contains information about books |
| Ratings.csv | Contains ratings given by users  |
| Users.csv   | Contains user information        |

---

# 💡 Future Improvements

Possible enhancements for this project:

- Build a hybrid recommendation system
- Add evaluation metrics for recommendation quality

---

# 👩‍💻 Author

Ananya Pappu

---

# Demo Link
https://book-rec-sys.streamlit.app/
