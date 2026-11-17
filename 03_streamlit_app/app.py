"""
Streamlit Application Entry Point: UniCompanion Evaluation Study.

This is the main root file (`app.py`) for the UniCompanion Streamlit application. 
It acts as the central router and global state manager for the entire experiment. 

Key responsibilities:
1. Orchestrating the Multi-Page App (MPA) navigation flow while hiding the sidebar 
   to force linear progression.
2. Managing the global participant counter via Google Sheets to track total visitors.
3. Pre-loading the evaluation dataset (`questions_data.json`) and randomly shuffling 
   the presentation order of the 4 experimental pipelines (Semantic Low/High, 
   Graph Low/High) to counteract sequence bias (counterbalancing).
4. Initializing the massive `st.session_state` payload that will accumulate the 
   user's interaction data as they navigate through the pages.
"""

import streamlit as st
import pandas as pd
import random
import uuid
import json
from streamlit_gsheets import GSheetsConnection

# 1. Page Config - Must be the first Streamlit command executed
st.set_page_config(page_title="UniCompanion Evaluation", page_icon="🤖", layout="centered")

# --- GLOBAL COUNTER FUNCTION ---
def get_and_increment_participant_counter():
    """
    Retrieves the global participant count from a Google Sheet and increments it.
    
    This counter acts as the `global_participant_sequence` and is crucial for 
    the round-robin question assignment logic located in `4_role_reminder.py`, 
    ensuring that incoming participants are served different slices of the dataset.
    
    Returns:
        int: The updated global participant index.
    """
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        # Fetch current count
        counter_data = conn.read(worksheet="Counter", ttl=0)
        if counter_data is not None and not counter_data.empty and "count" in counter_data.columns:
            val = counter_data["count"].iloc[0]
            count = int(val) if pd.notna(val) else 0
        else:
            count = 0
    except Exception as e:
        print(f"Counter read error: {e}")
        count = 0

    # Increment and write back to the database    
    new_data = pd.DataFrame([{"count": count + 1}])
    conn.update(worksheet="Counter", data=new_data)
    return count

# --- INITIALIZE STATE ---
# This block only runs once per participant session when they first land on the app
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    # Generate a unique alphanumeric ID to anonymously track the participant across files
    st.session_state.participant_id = str(uuid.uuid4())[:8]
    st.session_state.participant_index = get_and_increment_participant_counter()
    
    # State tracking for the block evaluation loops
    st.session_state.current_run = 1                        
    st.session_state.current_q_index = 0                    
    
    # Load JSON Dataset
    try:
        with open("questions_data.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)

            # The 4 experimental conditions being evaluated
            active_blocks = [
                "Semantic_Low_Explainable", 
                "Semantic_High_Explainable",
                "Graph_Low_Explainable",
                "Graph_High_Explainable"
            ]

            # Filter the raw data to only include the active blocks
            st.session_state.question_pools = {
                k: v for k, v in raw_data.items() if k in active_blocks
            }
    except FileNotFoundError:
        st.error("Error: 'questions_data.json' not found.")
        st.stop()

    # Randomize the presentation order of the 4 pipelines for this specific participant
    # This counterbalances the experiment to prevent sequence effects (e.g., fatigue)
    st.session_state.block_order = random.sample(list(st.session_state.question_pools.keys()), len(st.session_state.question_pools))
    
    st.session_state.active_pipeline = None
    st.session_state.active_questions = [] 

    # Initialize the master data payload that will be serialized at the end of the study
    st.session_state.responses = {
        "participant_id": st.session_state.participant_id,
        "global_participant_sequence": st.session_state.participant_index,
        "consent_given": False,
        "demographics": {},
        "block_presentation_order": st.session_state.block_order,
        "block_evaluations": {
            block_name: [] for block_name in st.session_state.block_order
        }
    }

# --- DEFINE PAGES & NAVIGATION ---
# Registering the sub-pages using Streamlit's native multi-page routing
pages = [
    st.Page("pages/1_consent.py", title="Consent Form"),
    st.Page("pages/2_demographics.py", title="Demographics"),
    st.Page("pages/3_instructions.py", title="Instructions"),
    st.Page("pages/4_role_reminder.py", title="Role Reminder"),
    st.Page("pages/5_question_eval.py", title="System Evaluation"),
    st.Page("pages/5b_bounce.py", title="Loading..."),
    st.Page("pages/6_pipeline_survey.py", title="Survey"),
    st.Page("pages/7_final.py", title="Finish")
]

# Position="hidden" disables the left sidebar menu. 
# This forces participants to follow the linear flow established by the buttons on each page.
pg = st.navigation(pages, position="hidden")
pg.run()
