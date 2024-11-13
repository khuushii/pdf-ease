import streamlit as st
from sqlalchemy import create_engine, Column, String, Integer, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pdfProcessing import pdfSection
from audioProcessing import audioSection
from videoProcessing import videoSection
from imageProcessing import imageSection

# Define SQLAlchemy models
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

# Connect to the database
DATABASE_URL = "sqlite:///users.db"
engine = create_engine(DATABASE_URL, echo=True)
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session_db = Session()

# Access session state
session_state = st.session_state

# Initialize session state if not exists
if not hasattr(session_state, "logged_in"):
    session_state.logged_in = False

def login(username, password):
    user = session_db.query(User).filter_by(username=username, password=password).first()
    if user:
        return True
    return False

def signup(username, password):
    existing_user = session_db.query(User).filter_by(username=username).first()
    if existing_user:
        st.error("Username already exists. Please choose a different username.")
    else:
        new_user = User(username=username, password=password)
        session_db.add(new_user)
        session_db.commit()
        st.success("Sign Up Successful! You can now login.")

def main():
    st.title("📄 pdfEASE 📄")

    if session_state.logged_in:
        st.subheader("Choose a section:")
        selected_section = st.radio(
            "Select an option",
            ["PDF Section","Image Section" ,"Audio Section", "Video Section"]
        )

        if selected_section == "PDF Section":
            pdfSection()
        elif selected_section == "Image Section":
            imageSection()
        elif selected_section == "Audio Section":
            audioSection()
        elif selected_section == "Video Section":
            videoSection()
        
    else:
        st.subheader("Login")
        username = st.text_input("Username:")
        password = st.text_input("Password:", type='password')

        if st.button("Login"):
            if login(username, password):
                session_state.logged_in = True
                st.success("Login Successful!")

        st.sidebar.subheader("Sign Up")
        new_username = st.sidebar.text_input("New Username:")
        new_password = st.sidebar.text_input("New Password:", type='password')

        if st.sidebar.button("Sign Up"):
            signup(new_username, new_password)

if __name__ == "__main__":
    main()
