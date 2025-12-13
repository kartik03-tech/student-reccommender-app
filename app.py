import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- DATABASE CONNECTION ----------
def get_db_connection():
    conn = sqlite3.connect("courses.db", check_same_thread=False)
    return conn


def load_data():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM courses", conn)
    conn.close()
    return df


def main():
    st.title("Course Recommendation System")

    # ✅ Menu (View Backend added)
    menu = ["Home", "Recommend Courses", "View Backend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Load data from SQL database
    df = load_data()

    # Combine text features
    df["combined_features"] = (
        df["coursetitle"] + " " +
        df["subject"] + " " +
        df["level"]
    )

    # ---------- HOME ----------
    if choice == "Home":
        st.subheader("Home")
        st.dataframe(df.head(10))

    # ---------- RECOMMEND ----------
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

                similarity_scores = cosine_similarity(
                    search_vector, vectors
                ).flatten()

                df["similarity"] = similarity_scores
                recommendations = df.sort_values(
                    by="similarity", ascending=False
                ).head(num_of_rec)

                st.success("Recommended Courses")

                for _, row in recommendations.iterrows():
                    st.markdown(f"""
                    ### 📘 {row['coursetitle']}
                    **Subject:** {row['subject']}  
                    **Level:** {row['level']}  
                    **Price:** ${row['price']}  
                    **Subscribers:** {row['num_subscribers']}
                    ---
                    """)

            else:
                st.warning("Please enter a search term")

    # ---------- VIEW BACKEND ----------
    elif choice == "View Backend":
        st.subheader("📦 Backend: SQLite Database")

        conn = get_db_connection()
        courses_df = pd.read_sql("SELECT * FROM courses", conn)
        st.dataframe(courses_df)
        conn.close()

    # ---------- ABOUT ----------
    else:
        st.subheader("About")
        st.write(
            "This application recommends courses using an SQL backend (SQLite) "
            "and cosine similarity-based content filtering."
        )


if __name__ == "__main__":
    main()
