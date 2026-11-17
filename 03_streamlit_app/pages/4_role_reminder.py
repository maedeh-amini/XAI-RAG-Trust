"""
Streamlit Page 4: Role Reminder & Block Initialization.

This page immerses the participant in the evaluation scenario (acting as a 
Cognitive Science student at UOS testing an Examination Office chatbot). 

Crucially, this script handles the complex logic for initializing the next evaluation 
block. It uses the `participant_index` to offset and assign a unique subset of 
questions from the pre-loaded question pools for each user, ensuring broad dataset 
coverage across the entire study while preventing duplicate questions within a session.
"""

import streamlit as st

# --- BASIC SESSION GUARD ---
# Ensure the participant has formally consented before accessing this page.
if "responses" not in st.session_state or "consent_given" not in st.session_state.responses:
    st.switch_page("pages/1_consent.py")

# Add the visual title to the actual page
st.title("🤖 UniCompanion")


# Advance progress bar to 30%
st.progress(0.3)
col_meta1, col_meta2 = st.columns([1, 1])
with col_meta1:
    st.caption(f"🆔 **Participant:** ID-{st.session_state.participant_id}")
st.markdown("---")

def initialize_next_block():
    """
    Initializes the specific RAG pipeline and question set for the upcoming evaluation block.
    
    This function implements the round-robin question assignment strategy. It calculates 
    a unique offset based on the participant's global index and the current block number, 
    ensuring that each participant evaluates a distinct slice of the dataset without overlaps.
    """
    # Identify which of the 4 pipelines (Semantic Low/High, Graph Low/High) is next
    current_block_name = st.session_state.block_order[st.session_state.current_run - 1]
    st.session_state.active_pipeline = current_block_name

    # Retrieve the full pool of questions specific to this pipeline
    pool = st.session_state.question_pools[current_block_name]
    pool_size = len(pool) 

    # Configuration: Each participant evaluates 3 questions per pipeline (12 total)
    questions_per_block = 3
    total_active_blocks = len(st.session_state.block_order)
    questions_per_participant = total_active_blocks * questions_per_block
    
    # Calculate starting index utilizing modulo arithmetic to loop back to the start of the pool if necessary
    participant_base_offset = (st.session_state.participant_index * questions_per_participant) % pool_size
    run_offset = (st.session_state.current_run - 1) * questions_per_block
    
    # Extract the exact 3 questions for this block
    active_qs = []
    for i in range(questions_per_block):
        target_index = (participant_base_offset + run_offset + i) % pool_size
        active_qs.append(pool[target_index])

    # Set the active session variables and route to the evaluation interface
    st.session_state.active_questions = active_qs
    st.session_state.current_q_index = 0
    st.switch_page("pages/5_question_eval.py")

st.subheader("🎯 Scenario & Role")
with st.container(border=True):
    # Sets the psychological context for the UX/Trust survey
    st.markdown("""
Imagine you are a Cognitive Science student at Osnabrück University trying to figure out your study regulations. The **Examination Office** has built *UniCompanion*, a new AI chatbot, to help answer your questions. 

Your critical task is to evaluate **4 different versions** of the AI chatbot and assess their responses for accuracy against official university regulations before the system goes live.

<br>
<div style="text-align: center;">
    <h4><strong>Please pay close attention to the details and evaluate each output carefully.</strong></h4>
</div>
""", unsafe_allow_html=True)
    
st.markdown("<br>", unsafe_allow_html=True)
# The button text dynamically updates based on which of the 4 blocks the user is starting
if st.button(f"Begin Block {st.session_state.current_run}", use_container_width=True):
    initialize_next_block()