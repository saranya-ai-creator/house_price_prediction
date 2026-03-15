import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dataset
data = {
    "area": [1000, 1500, 1800, 2400, 3000, 3500, 4000],
    "bedrooms": [2, 3, 3, 4, 4, 5, 5],
    "price": [300000, 400000, 450000, 550000, 650000, 700000, 800000]
}
df = pd.DataFrame(data)

# Features and target
X = df[["area", "bedrooms"]]
y = df["price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# User input
area = float(input("Enter house area: "))
bedrooms = int(input("Enter number of bedrooms: "))
prediction = model.predict([[area, bedrooms]])
print("Predicted house price:", prediction[0])

# Plot
plt.scatter(df["area"], df["price"])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price vs Area")
plt.show()
