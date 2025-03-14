import streamlit as st


def login_page():
    st.title("🔐 Login Page")
    st.text_input("Username")
    st.text_input("Password", type="password")

    if st.button("Login"):
        st.session_state["page"] = "chatapp"
        st.rerun()  # Forces reload


# Only run if executed directly
if __name__ == "__main__":
    login_page()
