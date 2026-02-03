import pandas as pd

# ----------------------------
#  LOAD DATA
# ----------------------------
df = pd.read_csv(
    r"C:\\Users\\devik\\OneDrive\\Desktop\\Coderzon\\Demand_Forecast\\Product_demand.csv",
    parse_dates=["Date"]
)

print("Initial shape:", df.shape)
import pandas as pd


df = df.dropna(subset=["Date"])

# ----------------------------
#  CLEAN ORDER_DEMAND
# ----------------------------

df["Order_Demand"] = (
    df["Order_Demand"]
    .astype(str)
    .str.strip()
    .astype(int)
)

# Demand cannot be negative or zero
df = df[df["Order_Demand"] > 0]

df = df.sort_values("Date")

# ----------------------------
#  CONVERT DAILY → MONTHLY
# ----------------------------

df["Month"] = df["Date"].dt.to_period("M")

monthly_df = (
    df.groupby(
        ["Category_Name", "Product_Name", "Month"],
        as_index=False
    )["Order_Demand"]
    .sum()
)

monthly_df["Month"] = monthly_df["Month"].dt.to_timestamp()

print("Monthly shape:", monthly_df.shape)

# ----------------------------
#  REMOVE VERY SPARSE PRODUCTS
# ----------------------------

MIN_MONTHS = 24

valid_products = (
    monthly_df.groupby("Product_Name")["Month"]
    .count()
    .loc[lambda x: x >= MIN_MONTHS]
    .index
)

monthly_df = monthly_df[
    monthly_df["Product_Name"].isin(valid_products)
]

print("After filtering sparse products:", monthly_df.shape)

# ----------------------------
#  FILL MISSING MONTHS (CRITICAL)
# ----------------------------

final_series = []

for (cat, prod), grp in monthly_df.groupby(["Category_Name", "Product_Name"]):

    grp = grp.sort_values("Month")

    full_idx = pd.date_range(
        start=grp["Month"].min(),
        end=grp["Month"].max(),
        freq="MS"
    )

    grp = (
        grp
        .set_index("Month")
        .reindex(full_idx)
        .fillna(0)
        .rename_axis("Month")
        .reset_index()
    )

    grp["Category_Name"] = cat
    grp["Product_Name"] = prod

    final_series.append(grp)

final_df = pd.concat(final_series, ignore_index=True)

# ----------------------------
#  FINAL SORT & TYPE CHECK
# ----------------------------

final_df = final_df.sort_values(
    ["Category_Name", "Product_Name", "Month"]
)

final_df["Order_Demand"] = final_df["Order_Demand"].astype(int)

# ----------------------------
#  SAVE CLEAN DATA
# ----------------------------

final_df.to_csv(
    "monthly_demand_data.csv",
    index=False
)

print(" Data preparation completed")
print(final_df.info())
print(final_df.shape)
