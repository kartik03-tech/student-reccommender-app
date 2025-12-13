import streamlit as st
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_db_connection():
    conn = sqlite3.connect("courses.db", check_same_thread=False)
    return conn

def load_data():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM courses", conn)
    conn.close()
    return df

def save_user_info(name, field_of_study, level, duration, mode):
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
    cursor.execute("""
        INSERT INTO users (name, field_of_study, level, duration, mode)
        VALUES (?, ?, ?, ?, ?)
    """, (name, field_of_study, level, duration, mode))
    conn.commit()
    conn.close()

def main():
    st.title("Course Recommendation System")
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    h1 { color: #00e5ff; text-align: center; }
    h2, h3 { color: #00e5ff; }
    input { border-radius: 10px !important; padding: 10px !important; }
    div.stButton > button {
        background-color: #00e5ff; color: black; border-radius: 10px;
        padding: 0.6em 1.2em; font-weight: bold; border: none;
    }
    div.stButton > button:hover { background-color: #00bcd4; color: white; }
    .course-card {
        background-color: rgba(255, 255, 255, 0.08);
        padding: 18px; border-radius: 15px; margin-bottom: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    }
    .highlight { color: #00e5ff; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    menu = ["Home", "Recommend Courses", "View Backend", "About"]
    st.sidebar.title("🎓 Navigation")
    choice = st.sidebar.radio("Go to", menu)

    df = load_data()
    df["combined_features"] = df["coursetitle"] + " " + df["subject"] + " " + df["level"]

    if choice == "Home":
        st.subheader("Welcome! Please fill in your details")

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
            ["Less than 1 month", "1-3 months", "3-6 months", "6+ months"]
        )
        mode = st.radio(
            "Mode of Study",
            ["Online", "Offline", "Hybrid"]
        )

        if st.button("Submit Info"):
            if name.strip() == "":
                st.warning("Please enter your name")
            else:
                save_user_info(name, field_of_study, level, duration, mode)
                st.success(f"Hello {name}! Your info has been saved.")

        st.dataframe(df.head(10))

    elif choice == "Recommend Courses":
        st.subheader("Recommend Courses")

        search_term = st.text_input(
            "Search Course",
            placeholder="Enter a course name or topic (e.g. Python, ML, Web)"
        )

        num_of_rec = st.sidebar.number_input(
            "Number of Courses to Recommend", 4, 30, 7
        )

        if st.button("Recommend"):
            if search_term:
                count_vect = CountVectorizer(stop_words="english")
                vectors = count_vect.fit_transform(df["combined_features"])
                search_vector = count_vect.transform([search_term])

                similarity_scores = cosine_similarity(search_vector, vectors).flatten()
                df["similarity"] = similarity_scores
                recommendations = df.sort_values(by="similarity", ascending=False).head(num_of_rec)

                st.success("Recommended Courses")

                for _, row in recommendations.iterrows():
                    st.markdown(f"""
                    <div class="course-card">
                    <h4>📘 {row['coursetitle']}</h4>
                    <p><span class="highlight">Subject:</span> {row['subject']}</p>
                    <p><span class="highlight">Level:</span> {row['level']}</p>
                    <p><span class="highlight">Price:</span> ${row['price']}</p>
                    <p><span class="highlight">Subscribers:</span> {row['num_subscribers']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a search term")

    elif choice == "View Backend":
        st.subheader("📦 Backend: SQLite Database")
        conn = get_db_connection()
        courses_df = pd.read_sql("SELECT * FROM courses", conn)
        st.dataframe(courses_df)

        st.subheader("Registered Users")
        users_df = pd.read_sql("SELECT * FROM users", conn)
        st.dataframe(users_df)

        conn.close()

    else:
        st.subheader("About")
        st.write(
            "This application recommends courses using an SQL backend (SQLite) "
            "and cosine similarity-based content filtering."
        )

if __name__ == "__main__":
    main()

