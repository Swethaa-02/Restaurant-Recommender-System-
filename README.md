
# 🍽️ Restaurant Recommendation System

## 📌 Overview
This project is a **Content-Based Restaurant Recommendation System** built with Python and Jupyter Notebooks.  
It suggests restaurants to users based on their **cuisine preferences**, **budget range**, and other features.

The project contains three main notebooks:
1. **Cuisine.ipynb** → Data cleaning, preprocessing, and cuisine analysis.
2. **Prediction.ipynb** → Predictive modeling for restaurant ratings or popularity.
3. **Recommendation.ipynb** → Content-based filtering system for restaurant recommendations.

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





