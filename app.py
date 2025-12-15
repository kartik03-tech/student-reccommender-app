import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- CSV PATH -----------------
CSV_PATH = "course.csv"

# ----------------- LOAD COURSES -----------------
def load_courses():
    return pd.read_csv(CSV_PATH)

# ----------------- STREAMLIT APP -----------------
def main():
    st.set_page_config(page_title="Course Recommender", layout="centered")
    st.title("🎓 Course Recommendation System")

    # Load courses
    try:
        df = load_courses()
    except:
        st.error("courses.csv file not found.")
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

        st.success(f"Hello {name}! Here are your recommendations 👇")

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


