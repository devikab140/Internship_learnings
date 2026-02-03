# import necessary libraries

import pandas as pd
import numpy as np
import holidays
from datetime import datetime

#  LOAD RAW DATA
orders = pd.read_csv("C:/Users/devik/Downloads/Olist/olist_orders_dataset.csv")
order_items = pd.read_csv("C:/Users/devik/Downloads/Olist/olist_order_items_dataset.csv")
products = pd.read_csv("C:/Users/devik/Downloads/Olist/olist_products_dataset.csv")
category_translation = pd.read_csv("C:/Users/devik/Downloads/Olist/product_category_name_translation.csv")

#  SELECT REQUIRED COLUMNS
orders = orders[['order_id', 'order_purchase_timestamp', 'order_status']]
order_items = order_items[['order_id', 'product_id', 'price', 'freight_value', 'order_item_id']]
products = products[['product_id', 'product_category_name']]
category_translation = category_translation[
    ['product_category_name', 'product_category_name_english']
]

#  FILTER VALID SALES
orders = orders[orders['order_status'] == 'delivered']

#  MERGE DATASETS
data_df = pd.merge(order_items, orders, on='order_id', how='inner')
data_df = pd.merge(data_df, products, on='product_id', how='left')
data_df = pd.merge(
    data_df,
    category_translation,
    on='product_category_name',
    how='left'
)

#  RENAME COLUMNS
data_df.rename(
    columns={
        'product_category_name_english': 'product_category',
        'order_purchase_timestamp': 'order_date'
    },
    inplace=True
)

#  TYPECAST DATE
data_df['order_date'] = pd.to_datetime(data_df['order_date'])
data_df['date'] = pd.to_datetime(data_df['order_date'].dt.date)

# print(data_df['date'].dt.year.value_counts().sort_index())

#  CREATE QUANTITY COLUMN
data_df['quantity'] = 1
data_df.drop('price', axis=1, inplace=True)

data_df['product_category'].value_counts().sort_index()

#  MERGE SIMILAR PRODUCT CATEGORIES TO REDUCE DIMENSIONALITY
CATEGORY_MAPPING = {
    # Home appliances & comfort
    'home_appliances_2': 'home_appliances',
    'home_confort': 'home_comfort',
    'home_comfort_2': 'home_comfort',
    # Telephony
    'fixed_telephony': 'telephony',
    # Small appliances
    'small_appliances_home_oven_and_coffee': 'small_appliances',
    # Food
    'food_drink': 'food',
    # Construction tools (typos + grouping)
    'costruction_tools_garden': 'construction_tools_garden',
    'costruction_tools_tools': 'construction_tools_tools',
    # Books
    'books_general_interest': 'books',
    'books_imported': 'books',
    'books_technical': 'books',
    # Media
    'dvds_blu_ray': 'cds_dvds_musicals',
    # Fashion typo fix
    'fashio_female_clothing': 'fashion_female_clothing',
    # Security
    'signaling_and_security': 'security_and_services'
}

data_df['product_category'] = (
    data_df['product_category']
    .replace(CATEGORY_MAPPING)
)

#  REMOVE NULLS
data_df.dropna(inplace=True)
# print(data_df.isnull().sum())

#  Aggregate daily demand per category
daily_demand = (
    data_df
    .groupby(['date', 'product_category'])
    .agg(units_sold=('quantity', 'sum'))
    .reset_index()
)

print(daily_demand.head())

#  SORT CHRONOLOGICALLY
daily_demand.sort_values('date', inplace=True)

#  ENSURE DATE CONTINUITY PER CATEGORY
final_dfs = []

for cat in daily_demand['product_category'].unique():

    temp = daily_demand[daily_demand['product_category'] == cat].copy()
    temp.set_index('date', inplace=True)

    temp = temp.asfreq('D')
    temp.fillna({'units_sold': 0}, inplace=True)
    temp['product_category'] = cat
    final_dfs.append(temp.reset_index())

daily_demand = pd.concat(final_dfs, ignore_index=True)

# ADD CALENDAR FEATURES
daily_demand['day_of_week'] = daily_demand['date'].dt.dayofweek
daily_demand['week_of_year'] = daily_demand['date'].dt.isocalendar().week.astype(int)
daily_demand['month'] = daily_demand['date'].dt.month
daily_demand['year'] = daily_demand['date'].dt.year

daily_demand['is_weekend'] = daily_demand['day_of_week'].isin([5, 6]).astype(int)



# Get all years in your dataset
years = daily_demand['year'].unique()

# Create Brazilian holidays for all relevant years
br_holidays = holidays.Brazil(years=years)

daily_demand['date_only'] = daily_demand['date'].dt.date
daily_demand['is_holiday'] = daily_demand['date_only'].isin(br_holidays).astype(int)
daily_demand['holiday_name'] = daily_demand['date_only'].map(br_holidays).fillna('No_Holiday')
daily_demand.drop('date_only', axis=1, inplace=True)

# ----------------------------
# TRANSLATE HOLIDAY NAMES
# ----------------------------
holiday_translation = {
    'Ano Novo': 'New Year',
    'Carnaval': 'Carnival',
    'Sexta-feira Santa': 'Good Friday',
    'Tiradentes': 'Tiradentes Day',
    'Dia do Trabalho': 'Labour Day',
    'Corpus Christi': 'Corpus Christi',
    'Independência do Brasil': 'Independence Day',
    'Nossa Senhora Aparecida': 'Our Lady of Aparecida',
    'Finados': 'All Souls Day',
    'Proclamação da República': 'Republic Day',
    'Natal': 'Christmas',
    'No_Holiday': 'No_Holiday'  # Keep non-holidays
}

daily_demand['holiday_name_eng'] = daily_demand['holiday_name'].replace(holiday_translation)

# Drop original Portuguese column
daily_demand.drop(columns=['holiday_name'], inplace=True)

# -------------------
# ADD MAJOR SHOPPING FESTIVALS
# -------------------
festival_dates = {
    'Carnival': ['2017-02-27', '2018-02-12'],
    'Black_Friday': ['2017-11-24', '2018-11-23'],
    'Christmas': ['2017-12-25', '2018-12-25'],
    'New_Year': ['2017-01-01', '2018-01-01']
}

daily_demand['is_festival'] = 0
daily_demand['festival_name'] = 'None'

for fest, dates in festival_dates.items():
    dates = pd.to_datetime(dates)
    daily_demand.loc[daily_demand['date'].isin(dates), 'is_festival'] = 1
    daily_demand.loc[daily_demand['date'].isin(dates), 'festival_name'] = fest

# -------------------
# FINAL CLEANING
# -------------------
daily_demand.fillna(0, inplace=True)

# SAVE FINAL DATASET
daily_demand.to_csv(
    "daily_demand_prepared.csv",
    index=False
)

print(" Data preparation completed successfully!")
print(daily_demand.head())








#https://www.kaggle.com/datasets/felixzhao/productdemandforecasting/code/data    , https://github.com/kcngnn/Product-Demand-Forecasting/blob/master/Code%20to%20run%20forecast%20automatically.ipynb