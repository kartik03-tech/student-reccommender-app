import sqlite3
import pandas as pd

# ----------------- Load CSV -----------------
df = pd.read_csv("data/course.csv")

# ----------------- Create database -----------------
conn = sqlite3.connect("courses.db")
cursor = conn.cursor()

# ----------------- Store CSV into courses table -----------------
df.to_sql("courses", conn, if_exists="replace", index=False)

# ----------------- Create users table -----------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    field_of_study TEXT,
    level TEXT,
    duration TEXT,
    mode TEXT
)
""")

conn.commit()
conn.close()
print("Database created successfully with courses and users tables")
