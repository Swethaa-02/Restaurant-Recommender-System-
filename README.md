
# 🍽️ Restaurant Recommendation System

## 📌 Overview
This project is a **Content-Based Restaurant Recommendation System** built with Python and Jupyter Notebooks.  
It suggests restaurants to users based on their **cuisine preferences**, **budget range**, and other features.

The project contains three main notebooks:
1. **Cuisine.ipynb** → Data cleaning, preprocessing, and cuisine analysis.
2. **Prediction.ipynb** → Predictive modeling for restaurant ratings or popularity.
3. **Recommendation.ipynb** → Content-based filtering system for restaurant recommendations.

---
## 📜 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Workflow](#workflow)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Recommendation Example](#recommendation-example)
7. [License](#license)
8. [Contributing](#contributing)

---


## 📂 Project Structure
restaurant-recommendation/
│
├── Cuisine.ipynb
├── Prediction.ipynb
├── Recommendation.ipynb
├── Dataset.csv # Optional - upload if size < 100MB
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 📊 Dataset
- **Source:** [Zomato Dataset](https://www.kaggle.com/datasets) (or your provided dataset).
- **Features Used:**
  - `Restaurant Name`
  - `Cuisines`
  - `Average Cost for two`
  - `City`
  - `Has Table booking`
  - `Has Online delivery`
  - `Aggregate rating`

---

## 📌  Work Flow
 1️⃣ Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

 2️⃣ Load Dataset
df = pd.read_csv("Dataset.csv")

 3️⃣ Data Preprocessing
Handle missing values
df.fillna({'Cuisines': '', 'City': 'Unknown'}, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

Encode categorical columns
label_encoders = {}
for col in ['City', 'Has Table booking', 'Has Online delivery']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

 4️⃣ Feature Extraction for Recommendation
TF-IDF for cuisines
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Cuisines'])

Combine features (cost, city, online delivery, etc.)
numerical_features = df[['Average Cost for two', 'City', 'Has Table booking', 'Has Online delivery']].values
from scipy.sparse import hstack
combined_features = hstack([tfidf_matrix, numerical_features])

5️⃣ Content-Based Recommendation Function
def recommend_restaurants(cuisine_pref, budget, top_n=5):
    user_tfidf = tfidf.transform([cuisine_pref])
    user_features = hstack([user_tfidf, np.array([[budget, 0, 0, 0]])])
    similarity_scores = cosine_similarity(user_features, combined_features).flatten()
    indices = similarity_scores.argsort()[-top_n:][::-1]
    return df.iloc[indices][['Restaurant Name', 'Cuisines', 'Average Cost for two', 'Aggregate rating']]

6️⃣ Prediction Model (Example: Predict Aggregate Rating)
X = df[['Average Cost for two', 'City', 'Has Table booking', 'Has Online delivery']]
y = df['Aggregate rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

 7️⃣ Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

 8️⃣ Actual vs Predicted Table
results_df = pd.DataFrame({
    'Restaurant Name': df.iloc[y_test.index]['Restaurant Name'].values,
    'Actual Rating': y_test.values,
    'Predicted Rating': y_pred
})
results_df['Predicted Rating'] = results_df['Predicted Rating'].round(1)
print(results_df.head())

 Save table for README
results_df.head(5).to_csv('actual_vs_predicted.csv', index=False)

 9️⃣ Test Recommendation
print("\nSample Recommendations for 'Japanese', Budget = 500:")
print(recommend_restaurants("Japanese", 500))

---

## ⚙️ Installation
1. **Clone the repository**  

git clone https://github.com/yourusername/restaurant-recommendation.git
cd restaurant-recommendation
2.Install dependencies
pip install -r requirements.txt
3.Run Jupyter Notebook
jupyter notebook

---

## 🚀 Usage
- Open the .ipynb files in Jupyter Notebook or JupyterLab.

- Run Cuisine.ipynb to explore and clean the dataset.

- Run Prediction.ipynb to train the predictive model.

- Run Recommendation.ipynb to generate restaurant recommendations.

---

## 🧠 Recommendation System Approach
- Preprocessing: Handle missing values, encode categorical variables, and normalize text data.

- eature Engineering: Use TF-IDF Vectorization for cuisines and Label Encoding for other categorical features.

- Similarity Measure: Cosine similarity to find restaurants most similar to the user’s preferences.

---

## 📌 Example Recommendation
User Preference: Cuisine = Japanese, Budget = 500 INR
Recommendation Output:

Restaurant Name	Cuisines	Average Cost for two	Rating
Sushi House	Japanese	500	4.5
Tokyo Table	Japanese	450	4.3
Samurai Sushi	Japanese	520	4.4

---

## 📜 License
This project is licensed under the MIT License.

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.





