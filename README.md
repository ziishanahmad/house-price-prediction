
# House Price Prediction

## Overview
This project aims to develop a machine learning model to predict house prices using the California Housing dataset. The dataset contains various features like the number of rooms, the age of the house, the distance to employment centers, and more.

## Key Features
- **Data Preprocessing:** Loading and standardizing the dataset.
- **Model Building:** Training a Linear Regression model to predict house prices.
- **Model Evaluation:** Assessing model performance using Mean Squared Error (MSE) and R-squared (R2) score.
- **Model Visualization:** Visualizing actual vs predicted house prices.

## Installation

### Clone the Repository
To get started, clone this repository to your local machine using the following command:
```sh
git clone https://github.com/ziishanahmad/house-price-prediction.git
cd house-price-prediction
```

### Set Up a Virtual Environment
It is recommended to use a virtual environment to manage your dependencies. You can set up a virtual environment using `venv`:
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Required Libraries
Install the necessary libraries using `pip`:
```sh
pip install -r requirements.txt
```

## Usage

### Run the Jupyter Notebook
Open the Jupyter notebook to run the project step-by-step:
1. Launch Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
2. Open the `house_price_prediction.ipynb` notebook.
3. Run the cells step-by-step to preprocess the data, train the model, evaluate its performance, and visualize the results.

## Detailed Explanation of the Code

### Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

### Load the Dataset
```python
housing = fetch_california_housing()
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
housing_df['PRICE'] = housing.target
housing_df.head()
```

### Exploratory Data Analysis (EDA)
```python
housing_df.describe()

plt.figure(figsize=(12, 10))
sns.heatmap(housing_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(housing_df['PRICE'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
```

### Data Preprocessing
```python
X = housing_df.drop('PRICE', axis=1)
y = housing_df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Build the Model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### Evaluate the Model
```python
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.

## Acknowledgements
- The California Housing dataset is provided by the [StatLib repository](http://lib.stat.cmu.edu/datasets/).
- The developers of TensorFlow for their deep learning framework.

## Contact
For any questions or feedback, please contact:
- **Name:** Zeeshan Ahmad
- **Email:** ziishanahmad@gmail.com
- **GitHub:** [ziishanahmad](https://github.com/ziishanahmad)
- **LinkedIn:** [ziishanahmad](https://www.linkedin.com/in/ziishanahmad/)

