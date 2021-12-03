# import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential

# import tensorflow as tf
# from sklearn.metrics import mean_squared_error


# %matplotlib inline
# %%
# Import Data
df = pd.read_csv("DATA/fake_reg.csv")

# %%
# Explore data
sns.pairplot(df)

# %%
# Split data test/train
X = df[["feature1", "feature2"]].values
y = df["price"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=101
)


# %%
# Scale data Only X
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# %%
# Build network
model = Sequential()

model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))

model.add(Dense(1))

model.compile(optimizer="rmsprop", loss="mse")
# %%
# Train model
model.fit(X_train, y_train, epochs=250)


# %%
# Evaluate model
loss_df = pd.DataFrame(model.history.history)
loss_df.plot(figsize=(12, 8))

y_pred = model.predict(X_test)

pred_df = pd.DataFrame(y_test)
pred = pd.Series(
    y_pred.reshape(
        300,
    )
)
pred_df = pd.concat([pred_df, pred], axis=1)
pred_df.columns = ["y_test", "y_pred"]

# %%
sns.scatterplot(x="y_test", y="y_pred", data=pred_df)
mean_absolute_error(pred_df["y_test"], pred_df["y_pred"])
# %%
# Predict on new data
new_gem = [[998, 1000]]
new_gem = scaler.transform(new_gem)
model.predict(new_gem)
# %%
# Save/load model
model.save("fake_model.h5")
loaded_model = load_model("22-Deep Learning/fake_model.h5")
# %%

# %%
