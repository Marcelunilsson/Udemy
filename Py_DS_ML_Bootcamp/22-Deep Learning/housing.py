# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# import numpy as np


sns.set(rc={"figure.figsize": (12, 8)})
print("Reading housing data...")
df = pd.read_csv("DATA/kc_house_data.csv")
print("Done.")
# %%
print(df.isnull().sum())
# %%
print(df.describe().transpose())
# %%
sns.displot(df["price"], kde=True)
plt.show()
# %%
sns.countplot(df["bedrooms"])
plt.show()
# %%
print(df.corr()["price"].sort_values())
# %%
sns.scatterplot(x="price", y="sqft_living", data=df)
plt.show()
# %%
sns.boxplot(x="bedrooms", y="price", data=df)
plt.show()
# %%
print(df.columns)
# %%
sns.scatterplot(x="price", y="long", data=df)
plt.show()
# %%
sns.scatterplot(x="price", y="lat", data=df)
plt.show()
# %%
sns.scatterplot(x="long", y="lat", data=df, hue="price", palette="magma_r")
plt.show()
# %%
df.sort_values("price", ascending=False).head(20)
# %%
below_4_mil = df[df["price"] < 4e6]
# %%
sns.scatterplot(
    x="long",
    y="lat",
    data=below_4_mil,
    hue="price",
    palette="magma_r",
    edgecolor=None,
    alpha=0.2,
)
plt.show()
# %%
sns.boxplot(x="waterfront", y="price", data=df)
# %%
df = df.drop("id", axis=1)
# %%
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].apply(lambda x: x.year)
df["month"] = df["date"].apply(lambda x: x.month)
df["day_of_week"] = df["date"].apply(lambda x: x.day_of_week)
# %%
df.head(5)
# %%
sns.boxplot(x="month", y="price", data=df)
# %%
df.groupby("month").mean()["price"].plot()
# %%
sns.boxplot(x="year", y="price", data=df)

# %%
df.groupby("year").mean()["price"].plot()
# %%
sns.boxplot(x="day_of_week", y="price", data=df)
# %%
df.groupby("day_of_week").mean()["price"].plot()

# %%
df.columns
# %%
df = df.drop(["zipcode", "day_of_week", "month", "date"], axis=1)
# %%
df.columns
# %%
df["yr_renovated"].value_counts()
# %%
df["sqft_basement"].value_counts()

# %%
df = df[df["price"] <= 3e6]
# %%
X = df.drop("price", axis=1).values
y = df["price"].values
# %%

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# %%
# %%
scaler = MinMaxScaler()
# %%
X_train = scaler.fit_transform(X_train)
# %%
X_test = scaler.transform(X_test)

# %%

# %%
X_train.shape
# %%
model = Sequential()

model.add(Dense(18, activation="relu"))
model.add(Dense(18, activation="relu"))
model.add(Dense(18, activation="relu"))
model.add(Dense(18, activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
# %%
model.fit(
    x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=128, epochs=400
)
# %%
losses = pd.DataFrame(model.history.history)
# %%
losses.plot()
# %%
# %%
y_pred = model.predict(X_test)
# %%
mean_squared_error(y_test, y_pred)
# %%
mean_absolute_error(y_test, y_pred)
# %%
explained_variance_score(y_test, y_pred)
# %%
plt.scatter(y_test, y_pred)
plt.plot(y_test, y_test, "r")
# %%
single_house = df.drop("price", axis=1).iloc[0]
# %%
single_house = scaler.transform(single_house.values.reshape(-1, 18))
# %%
model.predict(single_house)
# %%
df.head(1)
# %%
