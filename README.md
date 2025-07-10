# House-Price-Prediction

# 🏡 House Price Prediction 

This project predicts the sale price of homes in Ames, Iowa using machine learning techniques. It is based on the **Kaggle competition** [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

---

## 📌 Problem Statement

Accurately predict the final price of each home using 79 explanatory variables describing (almost) every aspect of residential homes.

---

## 📂 Dataset

- **Source**: [Kaggle Competition Link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Training Data**: `train.csv` – contains the sale prices
- **Test Data**: `test.csv` – to predict for Kaggle submission
- **Submission Format**: `sample_submission.csv`

---

## 🔍 Workflow Summary

1. **Data Loading & Exploration**
   - Imported dataset
   - Visualized distribution of `SalePrice` using histograms, boxplots, Q-Q plots
   - Analyzed skewness and transformed target variable with log1p

2. **Data Preprocessing**
   - Removed ID column
   - Handled missing values
   - Converted categorical variables to numerical using one-hot encoding

3. **Model Building**
   - Used `TensorFlow Decision Forests` with `RandomForestModel`
   - Trained on processed data
   - Evaluated with out-of-bag RMSE

4. **Prediction & Submission**
   - Generated predictions on test set
   - Created `submission.csv` for Kaggle

---

## 📈 Model Performance

- ✅ **Final RMSE (Out-of-Bag)**: *Add your value here*
- 📤 Submitted to Kaggle: [link if available]

---

## 🛠 Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- TensorFlow & TensorFlow Decision Forests

---

## 📁 Files in This Repo

| File | Description |
|------|-------------|
| `House_Price_Prediction.ipynb` | Jupyter Notebook with full workflow |
| `submission.csv` | Final prediction file for Kaggle |
| `README.md` | This summary file |

---

## 🧠 Future Improvements

- Try XGBoost, Ridge, or Lasso for comparison
- Use K-Fold Cross Validation
- Hyperparameter tuning

---

## 🙋‍♀️ Author

**Parimala Dharshini**  

[LinkedIn](https://www.linkedin.com/in/parimala-dharshini-903b4a271)  
