# 🏥 Medical Insurance Cost Prediction

## 📌 Project Overview

The **Medical Insurance Cost Prediction** project is a Machine Learning application that predicts the estimated medical insurance cost of an individual based on demographic and health-related information.

The project uses the **Medical Cost Personal Dataset** and applies data preprocessing, exploratory data analysis, feature encoding, and regression techniques to develop a predictive model. A **Polynomial Regression model with degree 2** is used as the final model because it captures nonlinear relationships between the input features and insurance charges.

A simple **Gradio web application** allows users to enter their information and receive an estimated insurance cost.

---

## 🎯 Problem Statement

Medical insurance costs can vary significantly between individuals based on factors such as age, BMI, smoking status, number of children, gender, and geographical region.

The objective of this project is to develop a Machine Learning model that can estimate an individual's medical insurance cost using these available features.

Such a prediction system can help demonstrate how historical insurance data can be used to identify patterns and generate cost estimates.

---

## 📊 Dataset

The project uses the **Medical Cost Personal Dataset**, commonly used for regression-based Machine Learning projects.

The dataset contains **1,338 records** and **7 columns**.

| Feature    | Description                        |
| ---------- | ---------------------------------- |
| `age`      | Age of the individual              |
| `sex`      | Gender of the individual           |
| `bmi`      | Body Mass Index                    |
| `children` | Number of dependent children       |
| `smoker`   | Whether the individual is a smoker |
| `region`   | Residential region                 |
| `charges`  | Medical insurance cost             |

### Target Variable

The target variable is:

**`charges`**

The model learns the relationship between the input features and `charges` to predict an estimated insurance cost for a new individual.

---

## 🔄 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Examined the dataset shape, columns, data types, and statistical information.
3. Checked for missing values.
4. Checked for duplicate records.
5. Identified categorical features.
6. Converted categorical variables into numerical values using **Pandas `get_dummies()`**.
7. Separated the dataset into input features (`X`) and target variable (`y`).
8. Split the data into training and testing sets using an **80/20 split**.

### Categorical Feature Encoding

The dataset contains three categorical variables:

- `sex`
- `smoker`
- `region`

These variables were converted into numerical dummy variables using Pandas:

```python
df_encoded = pd.get_dummies(
    df,
    columns=['sex', 'smoker', 'region'],
    drop_first=True,
    dtype=int
)
```

The `drop_first=True` option was used to remove one category from each categorical variable and avoid unnecessary redundant features.

After encoding, the categorical variables were represented as numerical columns that could be used by the Machine Learning models.

For example:

```text
sex → sex_male
smoker → smoker_yes
region → region_northwest
          region_southeast
          region_southwest
```

The target variable `charges` was then separated from the encoded features:

```python
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']
```

Finally, the data was divided into training and testing sets using an **80/20 train-test split**.

## 📈 Exploratory Data Analysis

Several visualizations and statistical analyses were performed to understand the dataset.

### Age vs Insurance Charges

The analysis showed that insurance charges generally tend to increase with age, although the relationship is not perfectly linear.

### BMI vs Insurance Charges

BMI has a positive relationship with insurance charges, but the relationship is weaker compared with some other variables.

### Smoking Status vs Charges

Smoking status showed a strong difference in insurance charges. Individuals classified as smokers had substantially higher average charges than non-smokers in this dataset.

These observations helped identify important patterns before building the predictive model.

---

## 🤖 Machine Learning Models

### 1. Multiple Linear Regression

A Linear Regression model was initially trained as a baseline model.

The model assumes a linear relationship between the input features and insurance charges.

### 2. Polynomial Regression

Polynomial Features with **degree 2** were then applied to the input data.

Polynomial transformation allows the model to capture additional relationships such as:

- `age²`
- `bmi²`
- `age × bmi`
- Other feature interactions

The transformed features were then used with Linear Regression.

The Polynomial Regression model achieved better performance on the test dataset than the baseline Linear Regression model.

---

## 📊 Model Evaluation

The final Polynomial Regression model was evaluated using the following metrics:

| Metric       |            Result |
| ------------ | ----------------: |
| **R² Score** |        **0.8825** |
| **MSE**      | **21,585,843.72** |
| **RMSE**     |      **4,646.06** |
| **MAE**      |      **2,867.32** |

### Interpretation

**R² Score = 0.8825**

The model explains approximately **88.25% of the variation in insurance charges** in the test dataset.

**RMSE = 4,646.06**

The RMSE represents the typical magnitude of prediction error in the same units as the insurance charges.

**MAE = 2,867.32**

On average, the model's predictions differ from the actual values by approximately **2,867 units of the target variable**, based on the test set.

---

## 🖥️ Prediction Application

A simple interactive web application was developed using **Gradio**.

Users can enter:

- Age
- BMI
- Number of children
- Gender
- Smoking status
- Region

The application processes the input using the same preprocessing and polynomial transformation used during model training and then generates an estimated insurance cost.

### Application Features

- Clean and responsive interface
- User-friendly input controls
- Instant prediction
- Estimated insurance cost display
- Model performance information
- Uses the trained Polynomial Regression model

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Joblib**
- **Gradio**
- **Google Colab**
- **GitHub**

---

## 📁 Project Structure

```text
medical-insurance-cost-prediction/
│
├── app.py
├── medical_insurance_prediction.ipynb
├── insurance_model.pkl
├── polynomial_features.pkl
├── requirements.txt
├── README.md
```

### File Description

| File                                 | Purpose                                            |
| ------------------------------------ | -------------------------------------------------- |
| `app.py`                             | Gradio prediction application                      |
| `medical_insurance_prediction.ipynb` | Complete data analysis and model training notebook |
| `insurance_model.pkl`                | Saved trained Polynomial Regression model          |
| `polynomial_features.pkl`            | Saved Polynomial Features transformer              |
| `requirements.txt`                   | Required Python libraries                          |
| `README.md`                          | Project documentation                              |

---

## ⚙️ How to Run the Project

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd medical-insurance-cost-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

The Gradio application will provide a local web interface where users can enter their information and obtain an estimated insurance cost.

---

## ⚠️ Limitations

This project has several limitations:

- The dataset contains only **1,338 records**.
- The dataset may not represent the population or insurance pricing practices of all regions.
- Predictions are estimates and should not be treated as actual insurance quotations.
- Important real-world factors may not be included in the dataset.
- Model performance depends on the quality and distribution of the available data.
- The reported evaluation metrics are based on a particular train-test split.

---

## 🔮 Future Improvements

Possible future improvements include:

- Testing additional regression algorithms.
- Hyperparameter tuning.
- Applying cross-validation.
- Using a larger and more diverse dataset.
- Adding prediction confidence or uncertainty estimates.
- Improving the user interface.
- Deploying the application as a production web service.
- Adding model monitoring and periodic retraining.

---

## 👨‍💻 Author

**Ahmad Imran**

BSCS Student
University of Management and Technology (UMT), Lahore, Pakistan

---

## 📚 Purpose

This project was developed for **academic and educational purposes** to demonstrate the complete Machine Learning workflow, including data preprocessing, exploratory data analysis, regression modeling, evaluation, model serialization, and deployment.

---

## 📄 License

This project is intended for educational and academic purposes.
