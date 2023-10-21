###############################################################
# Customer Segmentation with RFM
###############################################################

###############################################################
# Business Problem
###############################################################

###############################################################
# Data Understanding
###############################################################

import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width',1000)

df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()

df.head()
df["order_channel"].value_counts()
df["last_order_channel"].value_counts()

df.head(10)
df.columns
df.shape
df.describe().T
df.isnull().sum()
df.info()

# Omnichannel customers shop both online and offline platforms.
# We create new variables for the total number of purchases and spending of each customer.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df['customer_value_total_ever_online'] + df["customer_value_total_ever_offline"]
df.head()

# By examining the variable types, we convert the type of variables that express date to date.

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

# We looked at the distribution of the number of customers in shopping channels, the total number of products purchased and total expenditures.

df.groupby("order_channel").agg({"master_id":"count",
                                 "order_num_total":"sum",
                                 "customer_value_total":"sum"})

# Let's list the top 10 customers with the most revenue.

df.sort_values("customer_value_total", ascending=False)[:10]

# Let's list the top 10 customers with the most orders.

df.sort_values("order_num_total", ascending=False).head(10)

# Let's functionalize the data preparation process.

def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe['customer_value_total_ever_online'] + dataframe["customer_value_total_ever_offline"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df

###############################################################
#  Calculating RFM Metrics
###############################################################

# Analysis date 2 days after the date of the last purchase in the dataset

df["last_order_date"].max()
analysis_date = dt.datetime(2021,6,1)

# A new rfm dataframe with customer_id, recency, frequnecy and monetary values

rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (analysis_date - x.max()).days,
                                   "order_num_total":"sum",
                                   "customer_value_total":"sum"})
rfm.columns = ["recency", "frequency", "monetary"]
rfm.head()

###############################################################
# Calculating RF and RFM Scores
###############################################################

# Convert Recency, Frequency and Monetary metrics into scores between 1-5 with the help of qcut
# and save these scores as recency_score, frequency_score and monetary_score

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"),5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'],5, labels=[1, 2, 3, 4, 5])

rfm.head()

# Express recency_score and frequency_score as a single variable and save as RF_SCORE

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
rfm.head()

# Expressing #recency_score and frequency_score and monetary_score as a single variable and saving as RFM_SCORE

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))
rfm.head()

###############################################################
#  Segmental Identification of RF Scores
###############################################################

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head()

###############################################################
# Time for action!
###############################################################

# Analyse the recency, frequnecy and monetary averages of the segments.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

#                          recency       frequency       monetary
#                        mean count      mean count     mean count
# segment
# about_to_sleep       113.79  1629      2.40  1629   359.01  1629
# at_Risk              241.61  3131      4.47  3131   646.61  3131
# cant_loose           235.44  1200     10.70  1200  1474.47  1200
# champions             17.11  1932      8.93  1932  1406.63  1932
# hibernating          247.95  3604      2.39  3604   366.27  3604
# loyal_customers       82.59  3361      8.37  3361  1216.82  3361
# need_attention       113.83   823      3.73   823   562.14   823
# new_customers         17.92   680      2.00   680   339.96   680
# potential_loyalists   37.16  2938      3.30  2938   533.18  2938
# promising             58.92   647      2.00   647   335.67   647

# With the help of RFM analysis, find the customers in the relevant profile for 2 cases and save the customer ids to csv.

# a. FLO is adding a new women's footwear brand to its organisation. The product prices of the brand are above the general customer preferences. For this reason, the brand
# Customers who are interested in promoting and selling products should be contacted exclusively. These customers must be loyal and
#It was planned to be shoppers from the # women category. Customers' id numbers to csv file new_brand_target_customer_id.cvs
#Save as

rfm = rfm.reset_index()
rfm = rfm.merge(df[["master_id", "interested_in_categories_12"]], how = 'left')
rfm.head()
rfm.shape

woman_cat_loyal_or_champion_customers = rfm.loc[(rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers") & (rfm["interested_in_categories_12"].str.contains("KADIN")),["master_id"]]
woman_cat_loyal_or_champion_customers.shape
woman_cat_loyal_or_champion_customers.head()

woman_cat_loyal_or_champion_customers["master_id"].to_csv("woman_cat_loyal_or_champion_customers_masterid.csv")

#b.Nearly 40% discount is planned for Men's and Children's products. Previously interested in categories related to this discount
#good customers but not to be lost customers who have not shopped for a long time, those who are asleep and new customers
#want to be specifically targeted. Save the ids of the customers in the appropriate profile to the csv file.

man_and_child_cat_cant_loose_or_about_to_sleep_customers = rfm.loc[(rfm["segment"] == "cant_loose") | (rfm["segment"] == "about_to_sleep") & (rfm["interested_in_categories_12"].str.contains("ERKEK")) | (rfm["interested_in_categories_12"].str.contains("COCUK"))]
man_and_child_cat_cant_loose_or_about_to_sleep_customers["master_id"].to_csv("man_and_child_cat_cant_loose_or_about_to_sleep_customers_masterid.csv")