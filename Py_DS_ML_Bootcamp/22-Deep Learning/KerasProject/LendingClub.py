# %%
import random

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential

# %%
# Bigger plots
sns.set(rc={"figure.figsize": (12, 8)})
# %%
# Data info
data_info = pd.read_csv(
    "/home/marcel/Udemy/Py_DS_ML_Bootcamp/22-Deep Learning/KerasProject/lending_club_info.csv",
    index_col="LoanStatNew",
)
# %%
# Load Data
df = pd.read_csv(
    "/home/marcel/Udemy/Py_DS_ML_Bootcamp/22-Deep Learning/KerasProject/lending_club_loan.csv"
)
# %%
sns.countplot(x="loan_status", data=df)  # slightly skewed data
# %%
sns.histplot(x="loan_amnt", data=df, bins=35)
# %%
df.corr()
# %%
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
# %%
sns.scatterplot(x="installment", y="loan_amnt", data=df)
# %%


def feat_info(str):
    print(data_info.loc[str]["Description"])


# %%
sns.boxplot(x="loan_status", y="loan_amnt", data=df)
# %%
df.groupby("loan_status")["loan_amnt"].describe()
# %%
sns.countplot(x="grade", data=df, hue="loan_status")
# %%
sns.countplot(
    x="sub_grade",
    data=df,
    hue="loan_status",
    palette="magma",
    order=sorted(df["sub_grade"].unique()),
)
# %%
f_g = df[(df["grade"] == "G") | (df["grade"] == "F")]

sns.countplot(
    x="sub_grade",
    data=f_g,
    hue="loan_status",
    palette="magma",
    order=sorted(f_g["sub_grade"].unique()),
)
# %%
df["loan_repaid"] = df["loan_status"].map({"Fully Paid": 1, "Charged Off": 0})
# %%
df.corr()["loan_repaid"].sort_values().drop("loan_repaid").plot(kind="bar")
# %%
100 * df.isnull().sum() / len(df)
# %%
feat_info("emp_title")
# %%
feat_info("emp_length")
# %%
df["emp_title"].nunique()
# %%
df = df.drop("emp_title", axis=1)
# %%
sorted(df["emp_length"].dropna().unique())
# %%
order = [
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
]


sns.countplot(data=df, x="emp_length", order=order, hue="loan_status")
# %%
emp_co = (
    df[df["loan_status"] == "Charged Off"].groupby("emp_length").count()["loan_status"]
)
emp_tot = df.groupby("emp_length").count()["loan_status"]

emp_percent = emp_co / emp_tot
# %%
emp_percent.plot(kind="bar")
# %%
df = df.drop("emp_length", axis=1)
# %%
df = df.drop("title", axis=1)
# %%
df["mort_acc"].value_counts()
# %%
df.corr()["mort_acc"].sort_values()
# %%
fill_ma = df.groupby("total_acc").mean()["mort_acc"]
# %%


def fill(m_acc, t_acc):
    if np.isnan(m_acc):
        return fill_ma[t_acc]
    else:
        return m_acc


# %%
df["mort_acc"] = df.apply(lambda x: fill(x["mort_acc"], x["total_acc"]), axis=1)
# %%
df = df.dropna()
# %%
df.select_dtypes(["object"]).columns
# %%
df["term"] = df["term"].apply(lambda x: int(x[:3]))
# %%
df = df.drop("grade", axis=1)
# %%
subgrade_dummies = pd.get_dummies(df["sub_grade"], drop_first=True)
# %%
df = pd.concat([df.drop("sub_grade", axis=1), subgrade_dummies], axis=1)
# %%


def dummies(df, feat):
    dummies = pd.get_dummies(df[feat], drop_first=True)
    return pd.concat([df.drop(feat, axis=1), dummies], axis=1)


# %%
for feat in [
    "verification_status",
    "application_type",
    "initial_list_status",
    "purpose",
]:
    df = dummies(df, feat)

# %%
df["home_ownership"] = df["home_ownership"].replace(["NONE", "ANY"], "OTHER")
# %%
df = dummies(df, "home_ownership")
# %%
df["zip_code"] = df["address"].apply(lambda x: x[-5:])
# %%
df = dummies(df, "zip_code")
df = df.drop("address", axis=1)
# %%
df = df.drop("issue_d", axis=1)
# %%
df["earliest_cr_year"] = df["earliest_cr_line"].apply(lambda x: int(x[-4:]))
df = df.drop("earliest_cr_line", axis=1)
# %%
df = df.drop("loan_status", axis=1)
# %%
X = df.drop("loan_repaid", axis=1)
y = df["loan_repaid"]
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)
# %%
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
model = Sequential()


model.add(Dense(79, activation="relu"))
model.add(Dropout(0.30))

model.add(Dense(79, activation="relu"))
model.add(Dropout(0.15))

model.add(Dense(39, activation="relu"))
model.add(Dropout(0.30))

model.add(Dense(19, activation="relu"))
model.add(Dropout(0.30))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")
# %%
model.fit(
    x=X_train, y=y_train, epochs=50, batch_size=256, validation_data=(X_test, y_test)
)
# %%
model.save("full_data_project_model.h5")
# %%
losses = pd.DataFrame(model.history.history)
losses[["loss", "val_loss"]].plot()
# %%
y_pred = model.predict_classes(X_test)
cm = confusion_matrix(y_test, y_pred)
cd = ConfusionMatrixDisplay(confusion_matrix=cm)
cd.plot()
# %%
print(classification_report(y_test, y_pred))
# %%
random.seed(101)
random_ind = random.randint(0, len(df))

new_customer = df.drop("loan_repaid", axis=1).iloc[random_ind]
new_customer
# %%
nc_scal = scaler.transform(new_customer.values.reshape(1, 78))
new_pred = model.predict_classes(nc_scal)
# %%
