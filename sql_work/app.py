# from flask import Flask, jsonify
# from flask_sqlalchemy import SQLAlchemy

# app = Flask(__name__)

# # Connect to your existing MySQL database
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Devika%23513@localhost:3306/bank'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)

# # Customer table (existing one in DB)
# class Customer(db.Model):
#     __tablename__ = 'customer'
#     customer_id = db.Column(db.Integer, primary_key=True)
#     first_name = db.Column(db.String(100))
#     last_name = db.Column(db.String(100))
#     phone = db.Column(db.String(20))
#     email = db.Column(db.String(150))

#     def to_dict(self):
#         return {
#             "customer_id": self.customer_id,
#             "first_name": self.first_name,
#             "last_name": self.last_name,
#             "phone": self.phone,
#             "email": self.email
#         }

# @app.route('/')
# def home():
#     return jsonify({"message": "Welcome to Bank API — Read Only Mode"})

# # ✅ Fetch all customers (read-only)
# @app.route('/customers', methods=['GET'])
# def get_customers():
#     customers = Customer.query.all()
#     return jsonify([c.to_dict() for c in customers])

# # ✅ Fetch one customer by ID
# @app.route('/customers/<int:cid>', methods=['GET'])
# def get_customer(cid):
#     customer = Customer.query.get(cid)
#     if customer:
#         return jsonify(customer.to_dict())
#     return jsonify({"error": "Customer not found"}), 404

# if __name__ == '__main__':
#     app.run(debug=True)


#----------------------------------------------------------------
#all table in once

# from flask import Flask, jsonify
# from sqlalchemy import create_engine, MetaData, select

# app = Flask(__name__)

# # Connect to your MySQL database
# DB_URI = 'mysql+pymysql://root:Devika%23513@localhost:3306/bank'
# engine = create_engine(DB_URI)

# # Reflect existing database
# metadata = MetaData()
# metadata.reflect(bind=engine)

# @app.route('/')
# def home():
#     return jsonify({
#         "message": "Welcome to the Bank API",
#         "tables": list(metadata.tables.keys())
#     })

# @app.route('/table/<table_name>', methods=['GET'])
# def get_table_data(table_name):
#     if table_name not in metadata.tables:
#         return jsonify({"error": "Table not found"}), 404
    
#     table = metadata.tables[table_name]
#     with engine.connect() as conn:
#         result = conn.execute(select(table)).mappings().all()  # Important: .mappings() converts rows to dict
#         data = [dict(row) for row in result]
    
#     return jsonify(data)

# if __name__ == '__main__':
#     app.run(debug=True)





#--------------------------------------------------------
# #id wise getting

# from flask import Flask, jsonify
# from sqlalchemy import create_engine, text

# app = Flask(__name__)

# DB_URI = 'mysql+pymysql://root:Devika%23513@localhost:3306/bank'
# engine = create_engine(DB_URI)

# # Get table creation order
# def get_tables():
#     with engine.connect() as conn:
#         tables = [row[0] for row in conn.execute(text("""
#             SELECT TABLE_NAME
#             FROM INFORMATION_SCHEMA.TABLES
#             WHERE TABLE_SCHEMA = 'bank'
#             ORDER BY CREATE_TIME
#         """))]
#     return tables

# # Get columns with PK first
# def get_columns(table_name):
#     with engine.connect() as conn:
#         # Get primary key column
#         pk = conn.execute(text("""
#             SELECT COLUMN_NAME
#             FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
#             WHERE TABLE_SCHEMA = 'bank'
#               AND TABLE_NAME = :table
#               AND CONSTRAINT_NAME = 'PRIMARY'
#         """), {"table": table_name}).fetchone()
#         pk_col = pk[0] if pk else None

#         # Get all columns in creation order
#         all_cols = [row[0] for row in conn.execute(text("""
#             SELECT COLUMN_NAME
#             FROM INFORMATION_SCHEMA.COLUMNS
#             WHERE TABLE_SCHEMA = 'bank' AND TABLE_NAME = :table
#             ORDER BY ORDINAL_POSITION
#         """), {"table": table_name})]

#     # Ensure PK is first
#     if pk_col and pk_col in all_cols:
#         all_cols.remove(pk_col)
#         all_cols = [pk_col] + all_cols
#     return all_cols, pk_col

# @app.route('/')
# def home():
#     return jsonify({
#         "message": "Welcome to the Bank API",
#         "tables": get_tables()
#     })

# @app.route('/table/<table_name>', methods=['GET'])
# @app.route('/table/<table_name>/<int:pk_value>', methods=['GET'])
# def get_table_data(table_name, pk_value=None):
#     columns, pk_col = get_columns(table_name)
#     if not columns:
#         return jsonify({"error": "Table not found"}), 404

#     query = f"SELECT * FROM {table_name}"
#     params = {}

#     # Filter by primary key if provided
#     if pk_value is not None:
#         if not pk_col:
#             return jsonify({"error": "Primary key not defined for this table"}), 400
#         query += f" WHERE {pk_col} = :pk"
#         params['pk'] = pk_value
#     else:
#         # Sort by primary key by default
#         if pk_col:
#             query += f" ORDER BY {pk_col} ASC"

#     with engine.connect() as conn:
#         result = conn.execute(text(query), params).mappings().all()
#         data = [{col: row[col] for col in columns} for row in result]

#     return jsonify(data)

# if __name__ == "__main__":
#     app.run(debug=True)




#--------------------------------------
#sharable link
from flask import Flask, jsonify
from sqlalchemy import create_engine, text

app = Flask(__name__)

# -----------------------------
# DATABASE CONFIGURATION
# -----------------------------
DB_URI = 'mysql+pymysql://root:Devika%23513@localhost:3306/bank'
engine = create_engine(DB_URI)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

# Get tables in creation order
def get_tables():
    with engine.connect() as conn:
        tables = [row[0] for row in conn.execute(text("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'bank'
            ORDER BY CREATE_TIME
        """))]
    return tables

# Get columns with PK first and creation order
def get_columns(table_name):
    with engine.connect() as conn:
        # Primary key column
        pk = conn.execute(text("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = 'bank'
              AND TABLE_NAME = :table
              AND CONSTRAINT_NAME = 'PRIMARY'
        """), {"table": table_name}).fetchone()
        pk_col = pk[0] if pk else None

        # All columns in creation order
        all_cols = [row[0] for row in conn.execute(text("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'bank' AND TABLE_NAME = :table
            ORDER BY ORDINAL_POSITION
        """), {"table": table_name})]

    # Ensure PK is first
    if pk_col and pk_col in all_cols:
        all_cols.remove(pk_col)
        all_cols = [pk_col] + all_cols
    return all_cols, pk_col

# -----------------------------
# ROUTES
# -----------------------------

# Home route
@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Bank API",
        "tables": get_tables()
    })

# Fetch table data
@app.route('/table/<table_name>', methods=['GET'])
@app.route('/table/<table_name>/<int:pk_value>', methods=['GET'])
def get_table_data(table_name, pk_value=None):
    columns, pk_col = get_columns(table_name)
    if not columns:
        return jsonify({"error": "Table not found"}), 404

    query = f"SELECT * FROM {table_name}"
    params = {}

    # Filter by primary key if provided
    if pk_value is not None:
        if not pk_col:
            return jsonify({"error": "Primary key not defined for this table"}), 400
        query += f" WHERE {pk_col} = :pk"
        params['pk'] = pk_value
    else:
        # Sort by primary key by default
        if pk_col:
            query += f" ORDER BY {pk_col} ASC"

    with engine.connect() as conn:
        result = conn.execute(text(query), params).mappings().all()
        # Maintain column order
        data = [{col: row[col] for col in columns} for row in result]

    return jsonify(data)

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    # LAN access: listen on all IPs
    app.run(host="0.0.0.0", port=5000, debug=True)
