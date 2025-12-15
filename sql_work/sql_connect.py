import pymysql
import getpass
import re
import pandas as pd
print("Enter your MySQL connection details:")
host = input("Host (default: localhost): ").strip() or "localhost"
user = input("User (default: root): ").strip() or "root"
password = getpass.getpass("Password: ")
database = input("Database name (optional): ").strip() or None

SQL_FILE = "bank.sql"

# ---- Read SQL file ----
try:
    with open(SQL_FILE, "r", encoding="utf-8") as f:
        sql_script = f.read()
except FileNotFoundError:
    print(f" File '{SQL_FILE}' not found.")
    exit()

# ---- Remove comments ----
sql_script = re.sub(r'--.*\n', '', sql_script)
sql_script = re.sub(r'#.*\n', '', sql_script)
sql_script = re.sub(r'/\*[\s\S]*?\*/', '', sql_script)
commands = sql_script.split(';')

# ---- Connect to MySQL using context manager ,Connect using 'with' (auto close) ----
try:
    with pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        autocommit=True
    ) as connection:

        print("\n Connected successfully!\n")

        # Cursor also as context manager
        with connection.cursor() as cursor:

            # Execute all commands
            for command in commands:
                command = command.strip()
                if command:
                    try:
                        cursor.execute(command)
                        # print(f" Executed: {command[:60]}...")
                    except Exception as e:
                        print(f" Error executing: {command[:60]}...")
                        print("   →", e)
            alter_sql = "ALTER TABLE customer ADD COLUMN EXAMPLE VARCHAR(20)"
            cursor.execute(alter_sql)
            cursor.execute("UPDATE customer SET EXAMPLE = %s WHERE customer_id = %s", ("Test Example", 1))


            create_table_sql = """
            CREATE TABLE IF NOT EXISTS users (
                user_id INT AUTO_INCREMENT PRIMARY KEY,
                first_name VARCHAR(50),
                last_name VARCHAR(50),
                email VARCHAR(100)
            )
            """

            try:
                cursor.execute(create_table_sql)
                print("✅ Table 'users' created successfully.")
            except Exception as e:
                print("Error creating table:", e)


            cursor.execute("DROP TABLE IF EXISTS users")
            cursor.execute("ALTER TABLE customer DROP COLUMN EXAMPLE")
            # Show all tables
            cursor.execute("SHOW TABLES;")
            tables = [t[0] for t in cursor.fetchall()]
            print("\n Tables in database:")
            for table in tables:
                print("-", table)

#Preview first 5 rows of each table as a table
            for table in tables:
                print(f"\n First 5 rows of '{table}':")
                try:
                    df = pd.read_sql(f"SELECT * FROM {table} LIMIT 5;", connection)
                    print(df)  # nicely formatted table
                except Exception as e:
                    print(f" Could not read table '{table}':", e)


except Exception as e:
    print(" Connection or execution failed:", e)

print("\n Done! Connection and cursor auto-closed.")




