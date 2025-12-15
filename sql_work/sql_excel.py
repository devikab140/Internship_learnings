import pymysql
import pandas as pd
import getpass  # for hidden password input

# Ask user for MySQL connection details
host = input("Enter MySQL host (default: localhost): ").strip() or "localhost"
user = input("Enter MySQL user (default: root): ").strip() or "root"
password = getpass.getpass("Enter MySQL password: ")
database = input("Enter database name: ").strip()

# Connect to MySQL
connection = pymysql.connect(
    host=host,
    user=user,
    password=password,
    database=database,
    autocommit=True
)

print("Connected successfully!")

# Read investment table from MySQL (IDs 1-10)
df_sql = pd.read_sql("SELECT * FROM investment", connection)
# Read Excel/CSV data (IDs 11-20)
df_excel = pd.read_csv("C:/Users/devik/Downloads/investment.csv")  # make sure Excel file has investment_id 11-20
# ---- Clean up (optional, helps if Excel data has spaces) ----
df_excel.columns = df_excel.columns.str.strip()
df_excel = df_excel.applymap(lambda x: x.strip() if isinstance(x, str) else x)
combined_df = pd.concat([df_sql, df_excel], ignore_index=True)
# ---- Optional: remove duplicate rows (if any overlap) ----
combined_df = combined_df.drop_duplicates(subset="investment_id", keep="first")
# ---- Sort by investment_id ----
combined_df = combined_df.sort_values(by="investment_id")
# ---- Print result ----
print("\n Combined data from SQL and Excel:")
print(combined_df)


# # Take specific rows from SQL
# sample_sql = df_sql[df_sql['investment_id'].isin([1, 4, 8])]

# # Take specific rows from Excel
# sample_excel = df_excel[df_excel['investment_id'].isin([11, 17, 20])]

# # Combine
# combined_sample = pd.concat([sample_sql, sample_excel], ignore_index=True)

# #Optional: sort by investment_id
# combined_sample = combined_sample.sort_values(by="investment_id")

# print(combined_sample)

