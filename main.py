import streamlit as st

if "page" in st.session_state and st.session_state["page"] == "chatapp":
    from pages.chatapp import main as chatapp_main  # Use full path

    chatapp_main()  # Ensure `chatapp.py` has a `main()` function
else:
    from login import login_page

    login_page()
