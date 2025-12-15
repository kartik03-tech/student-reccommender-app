import streamlit as st
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- CSV PATH -----------------
CSV_PATH = "course.csv"
# ----------------- DATABASE -----------------
def get_connection():
    return sqlite3.connect("users.db", check_same_thread=False)

def create_users_table():
    conn = get_connection()
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

def save_user(name, field_of_study, level, duration, mode):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (name, field_of_study, level, duration, mode)
        VALUES (?, ?, ?, ?, ?)
    """, (name, field_of_study, level, duration, mode))
    conn.commit()
    conn.close()

# ----------------- LOAD COURSES -----------------
def load_courses():
    return pd.read_csv(CSV_PATH)

# ----------------- STREAMLIT APP -----------------
def main():
    st.set_page_config(page_title="Course Recommender", layout="centered")
    st.title("🎓 Course Recommendation System")

    # Create backend table
    create_users_table()

    # Load courses
    try:
        df = load_courses()
    except:
        st.error("course.csv file not found.")
        st.stop()

    # Combine text features
    df["combined_features"] = (
        df["coursetitle"] + " " + df["subject"] + " " + df["level"]
    )

    st.subheader("Welcome! Fill in your details to get recommendations")

    name = st.text_input("Your Name")

    field_of_study = st.selectbox(
        "Field of Study",
        ["Computer Science", "Data Science", "Business", "Arts", "Engineering", "Other"]
    )

    level = st.selectbox(
        "Level",
        ["Beginner", "Intermediate", "Advanced"]
    )

    duration = st.selectbox(
        "Duration Preference",
        ["<1 month", "1-3 months", "3-6 months", "6+ months"]
    )

    mode = st.radio(
        "Mode of Study",
        ["Online", "Offline", "Hybrid"]
    )

    if st.button("Recommend Courses"):
        if not name.strip():
            st.warning("Please enter your name")
            return

        # ✅ Save user to backend
        save_user(name, field_of_study, level, duration, mode)
        st.success("User information saved successfully ✅")

        # Recommendation logic
        search_term = f"{field_of_study} {level}"

        vectorizer = CountVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(df["combined_features"])
        search_vector = vectorizer.transform([search_term])

        similarity = cosine_similarity(search_vector, vectors).flatten()
        df["similarity"] = similarity

        recommendations = df.sort_values("similarity", ascending=False).head(5)

        st.subheader("📚 Recommended Courses")
        for _, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div style="font-size:12px; font-style:italic;">
                    📘 {row['coursetitle']}<br>
                    Subject: {row['subject']}<br>
                    Number of Lectures: {row['num_lectures']}<br>
                    Content Duration: {row['content_duration']} hours
                    <hr>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()

