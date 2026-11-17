"""
Streamlit Page 5b: Scroll-Reset Bounce Utility.

This is a hidden "trampoline" page used to solve a limitation in Streamlit's 
Single Page Application (SPA) architecture. When iterating through questions on 
the main evaluation page, Streamlit does not natively reset the browser's scroll 
position to the top. 

By routing the user to this dummy page and instantly routing them back to the 
evaluation interface, it forces a hard page re-render, naturally resetting the 
viewport to the top of the screen so the user doesn't have to manually scroll 
up for every new question.
"""

import streamlit as st

# --- STRICT SESSION GUARD ---
# Security check: Ensure the session state has actively assigned questions.
# If a user somehow lands here directly, redirect them to the consent page.
if "active_questions" not in st.session_state or len(st.session_state.active_questions) == 0:
    st.switch_page("pages/1_consent.py")


# Add the visual title to the actual page (briefly rendered in the DOM before the switch)
st.title("🤖 UniCompanion")


# Immediately bounce back to the question evaluation page. 
# Because it registers as a new page load, Streamlit natively scrolls to the top.
st.switch_page("pages/5_question_eval.py")