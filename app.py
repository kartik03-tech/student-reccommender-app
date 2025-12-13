import streamlit as st
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- DATABASE PATH -----------------
DB_PATH = "courses.db"

# ----------------- DATABASE FUNCTIONS -----------------
def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def ensure_users_table():
    """Ensure users table exists (IMPORTANT FIX)"""
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

    # ✅ ENSURE users table exists (CRITICAL LINE)
    ensure_users_table()

    # Load courses from existing DB
    try:
        df = load_courses()
    except Exception as e:
        st.error("Courses table not found in database.")
        st.stop()

    df["combined_features"] = (
        df["coursetitle"] + " " + df["subject"] + " " + df["level"]
    )

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
            return

        # Save user data safely
        try:
            save_user_info(name, field_of_study, level, duration, mode)
            st.success("User data saved successfully ✅")
        except Exception as e:
            st.error("Failed to save user data.")
            st.stop()

        # Recommendation logic
        search_term = f"{field_of_study} {level}"
        vectorizer = CountVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(df["combined_features"])
        search_vector = vectorizer.transform([search_term])
        similarity = cosine_similarity(search_vector, vectors).flatten()

        df["similarity"] = similarity
        recommendations = df.sort_values("similarity", ascending=False).head(5)

        st.subheader("Recommended Courses")
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
