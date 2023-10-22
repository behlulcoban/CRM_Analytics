##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma
##############################################################

##############################################################
# Business Problem
##############################################################

# FLO wants to establish a roadmap for its sales and marketing activities.
# To be able to plan for the medium to long-term future of the company, it is necessary to estimate the potential value that existing customers will bring to the company in the future.

###############################################################
# Dataset Information
###############################################################

# master_id: Unique customer ids
# order_channel : Which channel of the shopping platform was used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : Last shopping channel
# first_order_date : First shopping date of customer
# last_order_date : Last shopping date of customer
# last_order_date_online : Last online shopping date of customer
# last_order_date_offline : Last offline shopping date of customer
# order_num_total_ever_online : Total number of orders on online by customer
# order_num_total_ever_offline : Total number of orders on offline by customer
# customer_value_total_ever_offline : Total spending of customer on offline shopping
# customer_value_total_ever_online : Total spending of customer on online shopping
# interested_in_categories_12 : Categories that customer bought product in last 12 months

###############################################################
# Data Preperation
###############################################################

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.mode.chained_assignment = None

df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()
df.head()

#Define the outlier_thresholds and replace_with_thresholds functions needed to suppress outliers

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable]< low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable]> up_limit), variable] = round(up_limit, 0)

#if "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
#"customer_value_total_ever_online" columns have outliers, suppress them

columns = ["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)

df.describe().T

#Omnichannel means that customers shop from both online and offline platforms.
#We will create new variables for the total number of purchases and spend of each customer

df["order_num_total"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

#We will examine the variable types and convert the type of variables that express dates to date.

df.dtypes
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

##############################################
#Creating the CLTV Data Structure
##############################################

df["last_order_date"].max()
analysis_date = dt.datetime(2021,6,1)

# Let's create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.

cltv_df = pd.DataFrame()
cltv_df["cutomer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()

#Creating of BG/NBD, Gamma-Gamma Models and Calculation of CLTV

#Fit BG/NBD model

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

#We will estimate expected purchases from customers within 3 months and add exp_sales_3_month to cltv dataframe

cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

#We will estimate expected purchases from customers within 6 months and add exp_sales_6_month to cltv dataframe

cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

cltv_df[["exp_sales_3_month","exp_sales_3_month"]]

cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]
cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]


#Fit Gamma-Gamma model. We will estimate the average value of the customers and add it to the cltv dataframe as exp_average_value.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])
cltv_df.head()

#Calculating cltv for 6 month and adding cltv dataframe

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time = 6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

cltv_df.sort_values("cltv",ascending=False)[:10]

#Creating Segments by CLTV Value

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"],4,labels=["D", "C", "B", "A"])
cltv_df.head(20)

#comment: action recommendations for next 6 month
#The recency and age of the #A segment are lower than the other segments, and the frequencies are higher.
#Besides, the number of transactions it will make in 6 months and the average benefit it will bring seems higher.
#For this segment, which seems to provide the company with an average of 362,316 and a total of 1806505,089 revenues in a 6-month period,
#We can offer special campaigns that will increase the #purchasing rate, mentioning that there are special campaigns via e-mail and
#appealing to the customer, We need to take actions that will make you feel special and encourage shopping.

#B segment also goes close to the A segment. However, the C segment is not in a bad place in terms of shopping frequency and the benefit
#it will bring. Regular reminders can be made in order not to disturb the shopping routine, so as not to lose the C segment and preserve
#the situation. Categories of interest can be analyzed and information can be given in that direction.



# # Function for whole CLTV prediction process to improve functionality

def create_cltv_df(dataframe):

    # Veriyi Hazırlama
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # CLTV veri yapısının oluşturulması
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # # Gamma-Gamma Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # Cltv tahmini
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # CLTV segmentleme
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)


cltv_df.head(10)


