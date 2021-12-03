# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
df = df.drop(["price", "zipcode", "day_of_week", "month"], axis=1)
# %%
df.columns
# %%
df["yr_renovated"].value_counts()
# %%
