import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Data
data = {
    "hours_studied": [1, 2, 3, 4, 5],
    "score": [40, 50, 60, 70, 80]
}
df = pd.DataFrame(data)

X = df[["hours_studied"]]
y = df["score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Inference (CORRECT)
prediction = model.predict(
    pd.DataFrame([[3.5]], columns=["hours_studied"])
)

print(prediction)