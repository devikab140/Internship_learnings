from eralchemy import render_er

# Replace with your database credentials
db_connection = "mysql+pymysql://root:Devika#513@localhost:3306/bank"

# Output file
output_file = "bank_er_diagram.png"

# Generate ER diagram
render_er(db_connection, output_file)

print(f"✅ ER diagram generated successfully as '{output_file}'")
