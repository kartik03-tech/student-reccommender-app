import streamlit as st
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ----------------- DATABASE PATH (FIXED) -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "courses.db")

# ----------------- DATABASE FUNCTIONS -----------------
def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def create_courses_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coursetitle TEXT,
            subject TEXT,
            level TEXT,
            price REAL,
            num_subscribers INTEGER
        )
    """)
    conn.commit()
    conn.close()

def create_users_table():
    conn = get_db_connection()
    cursor = conn.cursor()
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

def insert_sample_courses():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM courses")
    if cursor.fetchone()[0] == 0:
        sample_data = [
            ("Python for Beginners", "Programming", "Beginner", 50, 1200),
            ("Data Science 101", "Data Science", "Beginner", 100, 850),
            ("Advanced AI", "Data Science", "Advanced", 200, 500),
            ("Business Analytics", "Business", "Intermediate", 150, 700),
            ("Art History Basics", "Arts", "Beginner", 30, 300)
        ]
        cursor.executemany("""
            INSERT INTO courses 
            (coursetitle, subject, level, price, num_subscribers) 
            VALUES (?, ?, ?, ?, ?)
        """, sample_data)
    conn.commit()
    conn.close()

def load_courses():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM courses", conn)
    conn.close()
    return df

def save_user_info(name, field_of_study, level, duration, mode):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (name, field_of_study, level, duration, mode)
        VALUES (?, ?, ?, ?, ?)
    """, (name, field_of_study, level, duration, mode))
    conn.commit()
    conn.close()

# ----------------- STREAMLIT APP -----------------
def main():
    st.set_page_config(page_title="Course Recommender", layout="centered")
    st.title("🎓 Course Recommendation System")

    # Database setup
    create_courses_table()
    create_users_table()
    insert_sample_courses()

    df = load_courses()
    df["combined_features"] = df["coursetitle"] + " " + df["subject"] + " " + df["level"]

    st.subheader("Welcome! Fill in your details to get recommendations")

    name = st.text_input("Your Name")
    field_of_study = st.selectbox(
        "Field of Study",
        ["Computer Science", "Data Science", "Business", "Arts", "Engineering", "Other"]
    )
    level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
    duration = st.selectbox(
        "Duration Preference",
        ["<1 month", "1-3 months", "3-6 months", "6+ months"]
    )
    mode = st.radio("Mode of Study", ["Online", "Offline", "Hybrid"])

    if st.button("Recommend Courses"):
        if not name.strip():
            st.warning("Please enter your name")
        else:
            save_user_info(name, field_of_study, level, duration, mode)
            st.success(f"Hello {name}! Your data is saved successfully ✅")

            search_term = f"{field_of_study} {level}"
            vectorizer = CountVectorizer(stop_words="english")
            vectors = vectorizer.fit_transform(df["combined_features"])
            search_vector = vectorizer.transform([search_term])
            similarity = cosine_similarity(search_vector, vectors).flatten()

            df["similarity"] = similarity
            recommendations = df.sort_values("similarity", ascending=False).head(5)

            for _, row in recommendations.iterrows():
                st.markdown(f"""
                **📘 {row['coursetitle']}**  
                Subject: {row['subject']}  
                Level: {row['level']}  
                Price: ${row['price']}  
                Subscribers: {row['num_subscribers']}  
                ---
                """)

if __name__ == "__main__":
    main()
