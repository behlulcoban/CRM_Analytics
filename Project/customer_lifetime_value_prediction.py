"""
CLTV Prediction with BG-NBD and Gamma-Gamma
CLTV Prediction with BG-NBD and Gamma-Gamma
-------------------------------------------
Business Problem:
UK based retail company sales and marketing
wants to determine a roadmap for its activities. The company
existing customers to be able to make medium and long term plans
the potential value they will provide to the company in the future
needs to be estimated.

Data Set Story:
The dataset, Online Retail II, was created by a UK-based retail company for the period 01/12/2009 - 09/12/2011.
between the two companies. The company's product catalog includes souvenirs and most of them
that the customer is a wholesaler.

Variables
8 Variables 541,909 Observations 45.6MB
InvoiceNo Invoice Number (If this code starts with C, it means that the transaction is canceled)
StockCode Product code (Unique for each product)
Description Product name
Product quantity (How many of the products in the invoices are sold)
InvoiceDate Invoice date
UnitPrice Invoice price ( Pounds )
CustomerID Unique customer number
Country Country name
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

df_ = pd.read_excel("dataset/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.info()
df.describe().T
df = df[~df["Invoice"].str.startswith("C", na=False)]
df = df[df["Quantity"] > 1]
df = df[df["Price"] > 0]
df["TotalPrice"] = df["Quantity"] * df["Price"]

def outlier_threshold(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[dataframe[variable] > up_limit, variable] = round(up_limit, 0)
    dataframe.loc[dataframe[variable] < low_limit, variable] = round(low_limit,0)

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# Task 1: Establishing BG-NBD and Gamma-Gamma Models and Forecasting 6-Month CLTV
# Step 1: Make a 6-month CLTV forecast for customers using data from 2010-2011.
# Step 2: Interpret and evaluate your results.
analysis_date = df["InvoiceDate"].max() + dt.timedelta(days=2)
cltv_df = df.groupby("Customer ID").agg({
    "InvoiceDate": [lambda date: (date.max() - date.min()).days, lambda date: (analysis_date - date.min()).days],
    "Invoice": lambda invoice: invoice.nunique(),
    "TotalPrice": "sum"})
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7
cltv_df = cltv_df[cltv_df["frequency"] > 1]
cltv_df["avg_monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])
cltv_df["exp_sales_6_month"] = bgf.predict(24, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["avg_monetary"])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["avg_monetary"])
cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                            cltv_df["frequency"],
                            cltv_df["recency"],
                            cltv_df["T"],
                            cltv_df["avg_monetary"],
                            time=6,
                            discount_rate=0.01,
                            freq="W")

# Task 2: CLTV Analysis of Different Time Periods
# Step 1: Calculate 1-month and 12-month CLTV for 2010-2011 UK customers.
# Step 2: Analyze the 10 people with the highest 1-month CLTV and the 10 people with the highest 12-month CLTV.
uk_customer_ids = (df[df["Country"] == "United Kingdom"]["Customer ID"]).unique()
uk_cltv_df = cltv_df[cltv_df.index.isin(uk_customer_ids)]
bgf.fit(uk_cltv_df["frequency"], uk_cltv_df["recency"], uk_cltv_df["T"])
ggf.fit(uk_cltv_df["frequency"], uk_cltv_df["avg_monetary"])
uk_cltv_df["cltv_1_month"] = ggf.customer_lifetime_value(bgf,
                                                         uk_cltv_df["frequency"],
                                                         uk_cltv_df["recency"],
                                                         uk_cltv_df["T"],
                                                         uk_cltv_df["avg_monetary"],
                                                         time=1,
                                                         discount_rate=0.01,
                                                         freq="W")
uk_cltv_df["cltv_12_month"] = ggf.customer_lifetime_value(bgf,
                                                         uk_cltv_df["frequency"],
                                                         uk_cltv_df["recency"],
                                                         uk_cltv_df["T"],
                                                         uk_cltv_df["avg_monetary"],
                                                         time=12,
                                                         discount_rate=0.01,
                                                         freq="W")
uk_cltv_df.sort_values("cltv_1_month", ascending=False).head(10)
uk_cltv_df.sort_values("cltv_12_month", ascending=False).head(10)

# Task 3: Segmentation and Action Recommendations
# Step 1: Divide all your customers into 4 groups (segments) according to the 6-month CLTV for 2010-2011 UK customers and
# Add the group names to the data set.
# Step 2: Make short 6-month action proposals to the management for 2 groups you will choose from among the 4 groups.
uk_cltv_df["cltv_6_month"] = ggf.customer_lifetime_value(bgf,
                                                         uk_cltv_df["frequency"],
                                                         uk_cltv_df["recency"],
                                                         uk_cltv_df["T"],
                                                         uk_cltv_df["avg_monetary"],
                                                         time=6,
                                                         discount_rate=0.01,
                                                         freq="W")
uk_cltv_df["segment"] = pd.qcut(uk_cltv_df["cltv_6_month"], 4, ["D", "B", "C", "A"])
uk_cltv_df.groupby("segment").agg(["mean", "sum", "count"])