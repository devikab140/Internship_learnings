# import os
# import getpass
# import pymysql
# import pandas as pd

# # ----------------- User inputs -----------------
# csv_path ="C:/Users/devik/Downloads/investment.csv"

# host = input("MySQL host (default localhost): ").strip() or "localhost"
# user = input("MySQL user (default root): ").strip() or "root"
# password = getpass.getpass("MySQL password: ")
# database = input("Database name (default Bank): ").strip() or "Bank"

# # ----------------- Connect to DB -----------------
# conn = pymysql.connect(host=host, user=user, password=password, database=database, autocommit=True)
# print(f"Connected to {database} at {host}")

# # ----------------- 1) Load investments from DB -----------------
# df_inv_sql = pd.read_sql("SELECT * FROM investment;", conn)
# print(f"Loaded {len(df_inv_sql)} investments from DB")

# # ----------------- 2) Load investments from CSV -----------------
# if not os.path.exists(csv_path):
#     raise SystemExit(f"CSV file not found: {csv_path}")

# df_inv_csv = pd.read_csv(csv_path)
# print(f"Loaded {len(df_inv_csv)} investments from CSV: {csv_path}")

# # ----------------- 3) Concatenate investments (no merges) -----------------
# df_inv_combined = pd.concat([df_inv_sql, df_inv_csv], ignore_index=True, sort=False)
# print(f"Combined investments rows: {len(df_inv_combined)}")

# # Ensure there is a customer_id column to link on
# if 'customer_id' not in df_inv_combined.columns:
#     raise SystemExit("Combined investments do not contain 'customer_id' column. The CSV must include customer_id.")

# # Normalize customer_id
# df_inv_combined['customer_id'] = df_inv_combined['customer_id'].astype(int)

# # ----------------- 4) Determine customers who have investments (combined) -----------------
# customers_with_investments = set(df_inv_combined['customer_id'].dropna().unique())
# print(f"Distinct customers with investments (combined): {len(customers_with_investments)}")

# if not customers_with_investments:
#     raise SystemExit("No customer_id found in the combined investments. Exiting.")

# # ----------------- 5) From DB find customers who have BOTH loan and locker (candidates) -----------------
# candidates_query = """
# SELECT DISTINCT c.customer_id
# FROM customer c
# WHERE EXISTS (SELECT 1 FROM loan l WHERE l.customer_id = c.customer_id)
#   AND EXISTS (SELECT 1 FROM locker lo WHERE lo.customer_id = c.customer_id)
# ;
# """
# df_candidates = pd.read_sql(candidates_query, conn)
# candidate_ids = set(df_candidates['customer_id'].astype(int).tolist())
# print(f"Customers who have BOTH loan and locker (candidates): {len(candidate_ids)}")

# # ----------------- 6) Intersection: customers who have loan+locker AND an investment -----------------
# selected_ids = sorted(list(candidate_ids.intersection(customers_with_investments)))
# print(f"Customers who have loan + locker + investment: {len(selected_ids)}")

# if len(selected_ids) == 0:
#     print("No customers satisfy all three (loan, locker, investment). Exiting.")
#     conn.close()
#     raise SystemExit

# # ----------------- 7) Fetch full details for selected customers -----------------
# # We fetch customer + account + loan + locker + staff + branch + sections info in one query.
# # Note: branch is joined using locker.branch_id (if available) or account.branch_id (fallback).
# ids_csv = ",".join(map(str, selected_ids))

# details_query = f"""
# SELECT
#   c.*,

#   -- account (a customer may have multiple accounts; this query will return one row per combination)
#   a.account_id   AS account_account_id,
#   a.account_num  AS account_num,
#   a.account_type AS account_type,
#   a.balance      AS account_balance,
#   a.branch_id    AS account_branch_id,

#   -- loan(s)
#   l.loan_id      AS loan_loan_id,
#   l.account_id   AS loan_account_id,
#   l.staff_id     AS loan_staff_id,
#   l.loan_type    AS loan_type,
#   l.principal_amount AS loan_principal,
#   l.interest_rate    AS loan_interest_rate,
#   l.emi_amount       AS loan_emi,
#   l.start_date       AS loan_start_date,
#   l.end_date         AS loan_end_date,
#   l.status           AS loan_status,

#   -- locker(s)
#   lo.locker_id   AS locker_locker_id,
#   lo.locker_num  AS locker_num,
#   lo.branch_id   AS locker_branch_id,
#   lo.locker_type AS locker_type,
#   lo.status      AS locker_status,

#   -- staff who handled the loan
#   s.staff_id     AS staff_id,
#   s.first_name   AS staff_first_name,
#   s.last_name    AS staff_last_name,
#   s.role         AS staff_role,
#   s.phone        AS staff_phone,
#   s.email        AS staff_email,
#   s.branch_id    AS staff_branch_id,
#   s.section_id   AS staff_section_id,

#   -- branch / section (branch linked from locker or account; section for branch and staff)
#   b.branch_id    AS branch_id,
#   b.branch_name  AS branch_name,
#   b.ifsc_code    AS branch_ifsc,
#   b.section_id   AS branch_section_id,

#   sec_branch.section_id   AS branch_section_section_id,
#   sec_branch.name         AS branch_section_name,

#   sec_staff.section_id    AS staff_section_section_id,
#   sec_staff.name          AS staff_section_name

# FROM customer c
# LEFT JOIN account a   ON a.customer_id = c.customer_id
# LEFT JOIN loan l      ON l.customer_id = c.customer_id
# LEFT JOIN staff s     ON l.staff_id = s.staff_id
# LEFT JOIN locker lo   ON lo.customer_id = c.customer_id

# -- try to get branch info (prefer locker.branch_id, else account.branch_id)
# LEFT JOIN branch b ON b.branch_id = COALESCE(lo.branch_id, a.branch_id)

# LEFT JOIN sections sec_branch ON b.section_id = sec_branch.section_id
# LEFT JOIN sections sec_staff  ON s.section_id = sec_staff.section_id

# WHERE c.customer_id IN ({ids_csv})
# ORDER BY c.customer_id;
# """

# df_details = pd.read_sql(details_query, conn)

# # ----------------- 8) Present results -----------------
# print("\n=== Summary ===")
# print(f"Total selected customers: {len(selected_ids)}")
# print(f"Rows returned by details query: {len(df_details)} (one row per combination of related child rows)")

# print("\n=== Selected customer IDs ===")
# print(selected_ids)

# print("\n=== Details preview (top 10 rows) ===")
# print(df_details.head(10).to_string(index=False))

# # If you want a one-row-per-customer view (no cartesian explosion), you can reduce it by grouping.
# # But you said no merging — so we show the raw details as returned above.

# conn.close()
# print("\nDone.")




import os, getpass, pymysql, pandas as pd

csv_path ="C:/Users/devik/Downloads/investment.csv"
host = input("MySQL host (default localhost): ").strip() or "localhost"
user = input("MySQL user (default root): ").strip() or "root"
pw = getpass.getpass("MySQL password: ")
db = input("Database name (default Bank): ").strip() or "Bank"

conn = pymysql.connect(host=host, user=user, password=pw, database=db, autocommit=True)

# 1) investments: DB + CSV
inv_db = pd.read_sql("SELECT * FROM investment", conn)
inv_csv = pd.read_csv(csv_path)
inv = pd.concat([inv_db, inv_csv], ignore_index=True, sort=False)
if 'customer_id' not in inv.columns: raise SystemExit("investments must have customer_id")
cust_with_inv = set(inv['customer_id'].dropna().astype(int).unique())

# 2) customers who have loan AND locker
cands = pd.read_sql("""
SELECT DISTINCT c.customer_id
FROM customer c
WHERE EXISTS(SELECT 1 FROM loan l WHERE l.customer_id=c.customer_id)
  AND EXISTS(SELECT 1 FROM locker lo WHERE lo.customer_id=c.customer_id)
""", conn)
cand_ids = set(cands['customer_id'].astype(int).tolist())

selected = sorted(cand_ids & cust_with_inv)
if not selected:
    print("No customers have loan+locker+investment"); conn.close(); raise SystemExit

ids_csv = ",".join(map(str, selected))

# 3) fetch full details (may return multiple rows per customer)
q = f"""
SELECT c.*, a.account_id,a.account_num,a.account_type,a.balance,
       l.loan_id,l.loan_type,l.principal_amount,l.status   AS loan_status,
       lo.locker_id,lo.locker_num,lo.locker_type,lo.status AS locker_status,
       s.staff_id,s.first_name AS staff_first_name,s.last_name AS staff_last_name,s.role,
       b.branch_id,b.branch_name,b.ifsc_code,
       sec_branch.name AS branch_section, sec_staff.name AS staff_section
FROM customer c
LEFT JOIN account a ON a.customer_id=c.customer_id
LEFT JOIN loan l ON l.customer_id=c.customer_id
LEFT JOIN locker lo ON lo.customer_id=c.customer_id
LEFT JOIN staff s ON l.staff_id=s.staff_id
LEFT JOIN branch b ON b.branch_id = COALESCE(lo.branch_id, a.branch_id)
LEFT JOIN sections sec_branch ON b.section_id = sec_branch.section_id
LEFT JOIN sections sec_staff ON s.section_id = sec_staff.section_id
WHERE c.customer_id IN ({ids_csv})
ORDER BY c.customer_id
"""
df = pd.read_sql(q, conn)
conn.close()

print(f"Selected customer IDs: {selected}")
print(f"Detail rows returned: {len(df)}")
print(df.head(10).to_string(index=False))
