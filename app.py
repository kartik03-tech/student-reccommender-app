import streamlit as st
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- DATABASE FUNCTIONS -----------------
def get_db_connection():
    conn = sqlite3.connect("courses.db", check_same_thread=False)
    return conn

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
        cursor.executemany(
            "INSERT INTO courses (coursetitle, subject, level, price, num_subscribers) VALUES (?, ?, ?, ?, ?)",
            sample_data
        )
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
    # Create users table if not exists
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
    # Insert user info
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

    # ----------------- DATABASE SETUP -----------------
    create_courses_table()
    insert_sample_courses()
    df = load_courses()
    df["combined_features"] = df["coursetitle"] + " " + df["subject"] + " " + df["level"]

    # ----------------- SIDEBAR -----------------
    st.sidebar.title("💡 About Us")
    st.sidebar.markdown("""
Welcome to **Course Recommender**!  

- Discover courses that match your **interests and level**.  
- Get **personalized recommendations** instantly.  
- Securely store your preferences in our **SQL database**.  
- Learn and grow with courses curated just for you!  
    """)

    # CSS Styling
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color: white; }
    h1,h2,h3 { color: #00e5ff; text-align: center; }
    div.stButton > button { background-color: #00e5ff; color: black; border-radius: 10px; padding: 0.6em 1.2em; font-weight: bold; border: none; }
    div.stButton > button:hover { background-color: #00bcd4; color: white; }
    .course-card { background-color: rgba(255,255,255,0.08); padding:18px; border-radius:15px; margin-bottom:15px; box-shadow:0px 4px 15px rgba(0,0,0,0.3);}
    .highlight { color: #00e5ff; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    # ----------------- USER INPUT -----------------
    st.subheader("Welcome! Fill in your details to get recommendations")

    name = st.text_input("Your Name")
    field_of_study = st.selectbox("Field of Study", ["Computer Science", "Data Science", "Business", "Arts", "Engineering", "Other"])
    level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
    duration = st.selectbox("Duration Preference", ["<1 month", "1-3 months", "3-6 months", "6+ months"])
    mode = st.radio("Mode of Study", ["Online", "Offline", "Hybrid"])

    if st.button("Recommend Courses"):
        if not name.strip():
            st.warning("Please enter your name")
        else:
            # Save user info to SQL
            save_user_info(name, field_of_study, level, duration, mode)
            st.success(f"Hello {name}! Here are some recommended courses:")

            # Content-based recommendation
            search_term = f"{field_of_study} {level} {mode} {duration}"
            count_vect = CountVectorizer(stop_words="english")
            vectors = count_vect.fit_transform(df["combined_features"])
            search_vector = count_vect.transform([search_term])
            similarity_scores = cosine_similarity(search_vector, vectors).flatten()
            df["similarity"] = similarity_scores
            recommendations = df.sort_values(by="similarity", ascending=False).head(7)

            if recommendations.empty:
                st.info("No courses found matching your preferences.")
            else:
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

    # ----------------- HIDDEN ADMIN PANEL -----------------
    admin_key = st.text_input("Enter admin key to view users", type="password")
    if admin_key == "SECRET123":  # Replace with your secret key
        st.subheader("👤 Users Table")
        conn = get_db_connection()
        users_df = pd.read_sql("SELECT * FROM users", conn)
        conn.close()
        st.dataframe(users_df)

if __name__ == "__main__":
    main()
