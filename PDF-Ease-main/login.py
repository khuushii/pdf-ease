import streamlit as st
from sqlalchemy import create_engine, Column, String, Integer, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from audioProcessing import audioSection
from videoProcessing import videoSection
from app import pdfSection

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
session = Session()

def login(username, password):
    user = session.query(User).filter_by(username=username, password=password).first()
    if user:
        return True
    return False

def signup(username, password):
    existing_user = session.query(User).filter_by(username=username).first()
    if existing_user:
        st.error("Username already exists. Please choose a different username.")
    else:
        new_user = User(username=username, password=password)
        session.add(new_user)
        session.commit()
        st.success("Sign Up Successful! You can now login.")

def main():
    st.title("Login/Sign Up Page")

    page = st.sidebar.radio("Select your action", ["Login", "Sign Up"])

    if page == "Login":
        st.subheader("Login")
        username = st.text_input("Username:")
        password = st.text_input("Password:", type='password')

        if st.button("Login"):
            if login(username, password):
                st.success("Login Successful!")

                with st.sidebar:
                    st.subheader("Choose a method of Talking")
                    radio_selection = st.radio(
                        "Select an option",
                        [" ","Talk to PDF", "Talk to Audio", "Talk to Video"]
                    )
                if radio_selection == " ":
                    st.write("no operation")
                elif radio_selection == "Talk to PDF":
                    pdfSection()
                elif radio_selection == "Talk to Audio":
                    audioSection()
                elif radio_selection == "Talk to Video":
                    videoSection()    

            else:
                st.error("Invalid Credentials. Please try again.")

    elif page == "Sign Up":
        st.subheader("Sign Up")
        new_username = st.text_input("New Username:")
        new_password = st.text_input("New Password:", type='password')

        if st.button("Sign Up"):
            signup(new_username, new_password)

if __name__ == "__main__":
    main()
