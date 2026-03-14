import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("retail_store_inventory.csv")

# Convert categorical columns to numeric
df = pd.get_dummies(df)

X = df.drop(["Units Sold"], axis=1)
y = df["Units Sold"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(prediction[:10])