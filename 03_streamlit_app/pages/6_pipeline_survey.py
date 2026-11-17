"""
Streamlit Page 6: Post-Block Pipeline Survey.

This page executes Phase 2 of the evaluation block: measuring user perception.
After a participant evaluates the RAG interaction logs for a specific pipeline 
(e.g., Graph High Explainability), they are routed here to rate the system's 
trustworthiness, reliability, and the utility of its explanations using a 
7-item Likert scale.

Critically, this script also manages the backend data serialization. When the 
final block is completed, it compiles all session state data (demographics, 
interaction scores, and surveys) into JSON strings and securely transmits 
the record to a central Google Sheets database.
"""

import streamlit as st
import pandas as pd
import json
from streamlit_gsheets import GSheetsConnection

# --- STRICT SESSION GUARD ---
# Security check: Ensure the core block sequencing variables exist. 
# If the user refreshes the page and wipes the state, redirect them to start over safely.
if "block_order" not in st.session_state or "current_run" not in st.session_state:
    st.switch_page("pages/1_consent.py")

# Add the visual title to the actual page
st.title("🤖 UniCompanion")

# Dynamic progress scaling to represent nearing completion of the study
adjusted_progress = 0.8
st.progress(min(adjusted_progress, 1.0))

col_meta1, col_meta2 = st.columns([1, 1])
with col_meta1:
    st.caption(f"🆔 **Participant:** ID-{st.session_state.get('participant_id', 'Unknown')}")
with col_meta2:
    st.caption(f"🧱 **Progress:** Block {st.session_state.current_run} / {len(st.session_state.block_order)}")

def save_responses_to_server():
    """
    Serializes the entire user session (demographics, permutations, evaluation 
    scores, and surveys) and appends it as a single row to a remote Google Sheet.
    
    Data is formatted as JSON strings within the columns to preserve the nested 
    dictionary structures required for later statistical parsing (e.g., ANOVA in R).
    """
    # Establish secure connection to Google Sheets via Streamlit secrets
    conn = st.connection("gsheets", type=GSheetsConnection)
    existing_data = conn.read(worksheet="ExperimentData", ttl=0)
    
    # Expected pipelines, aligning with the 4 experimental conditions
    chatbot_types = [
        "Semantic_Low_Explainable", "Semantic_High_Explainable",
        "Graph_Low_Explainable", "Graph_High_Explainable"
    ]

    # Construct base participant metadata
    row_data = {
        "participant_id": st.session_state.responses["participant_id"],
        "global_participant_sequence": st.session_state.responses["global_participant_sequence"],
        "demographics": json.dumps(st.session_state.responses["demographics"], ensure_ascii=False),
        "block_presentation_order": json.dumps(st.session_state.responses["block_presentation_order"], ensure_ascii=False),
        "consent_given": st.session_state.responses["consent_given"]
    }

    # Flatten the evaluation and survey arrays into the specific pipeline columns
    for chatbot in chatbot_types:
        if chatbot in st.session_state.responses["block_evaluations"]:
            eval_list = st.session_state.responses["block_evaluations"][chatbot]
            row_data[f"single_question_rating_{chatbot}"] = json.dumps(eval_list, ensure_ascii=False) if eval_list else "[]"
        else:
            row_data[f"single_question_rating_{chatbot}"] = "[]"
            
        if "block_surveys" in st.session_state.responses and chatbot in st.session_state.responses["block_surveys"]:
            survey_dict = st.session_state.responses["block_surveys"][chatbot]
            row_data[f"block_survey_{chatbot}"] = json.dumps(survey_dict, ensure_ascii=False)
        else:
            row_data[f"block_survey_{chatbot}"] = "{}"

    # Append the new record to the dataframe and push to Google Sheets       
    new_data_row = pd.DataFrame([row_data])
    updated_df = pd.concat([existing_data, new_data_row], ignore_index=True) if existing_data is not None else new_data_row
    conn.update(worksheet="ExperimentData", data=updated_df)

def submit_pipeline_survey(survey_scores):
    """
    Callback function executed when the survey form is submitted.
    Records the Likert scores and routes the application to either the 
    next evaluation block or the final completion page.
    """
    survey_data = {
        "block_sequence_position": st.session_state.current_run,
        "assigned_pipeline": st.session_state.active_pipeline,
        "survey_responses": survey_scores
    }
    
    if "block_surveys" not in st.session_state.responses:
        st.session_state.responses["block_surveys"] = {}
        
    # Save current block's survey to the global response dictionary
    st.session_state.responses["block_surveys"][st.session_state.active_pipeline] = survey_data
    
    # Logic Router: Proceed to next block OR save to server and finish
    if st.session_state.current_run < len(st.session_state.block_order):
        st.session_state.current_run += 1
        
        # Initialize next block routing logic (re-implementing the round-robin logic from page 4)
        current_block_name = st.session_state.block_order[st.session_state.current_run - 1]
        st.session_state.active_pipeline = current_block_name
        pool = st.session_state.question_pools[current_block_name]
        
        questions_per_block = 3
        total_active_blocks = len(st.session_state.block_order)
        questions_per_participant = total_active_blocks * questions_per_block
        
        participant_base_offset = (st.session_state.participant_index * questions_per_participant) % len(pool)
        run_offset = (st.session_state.current_run - 1) * questions_per_block
        
        active_qs = []
        for i in range(questions_per_block):
            target_index = (participant_base_offset + run_offset + i) % len(pool)
            active_qs.append(pool[target_index])
        
        st.session_state.active_questions = active_qs
        st.session_state.current_q_index = 0
        st.switch_page("pages/5_question_eval.py")
    else:
        # Final block completed: Push to remote database and route to finish
        save_responses_to_server()
        st.switch_page("pages/7_final.py")

st.subheader(f"📝 End of Block {st.session_state.current_run} Review")
st.write("Please indicate your level of agreement with the descriptions regarding the system you just evaluated:")
st.markdown("---")

# Custom CSS to ensure proper vertical alignment of Likert scale radio buttons
st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] { align-items: center !important; }
    </style>
""", unsafe_allow_html=True)

# Define the 7 psychometric evaluation statements (measuring trust, reliability, and XAI utility)
items = [
    {"text": "I trust the chatbot's responses.", "key": "trust_responses"},
    {"text": "The chatbot's response seemed reliable.", "key": "responses_reliable"},
    {"text": "I felt confident making a decision based on the chatbot's response.", "key": "confident_decision"},
    {"text": "I would use this chatbot to make a similar academic or study-related decision.", "key": "use_future_decisions"},
    {"text": "The explanation helped me understand the answer provided.", "key": "helped_understanding"},
    {"text": "The explanation made me trust the chatbot more.", "key": "explanation_increased_trust"},
    {"text": "I prefer getting a contextual explanation over just a direct quote from the study regulations.", "key": "prefer_contextual"}
]

likert_options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
current_survey_answers = {}

# Dynamically render the survey form
for idx, item in enumerate(items):
    st.markdown(f"**Item {idx+1}:** {item['text']}")
    dict_key = f"item{idx+1}"
    
    # By enforcing index=None, the participant must actively click a button, preventing default-bias
    choice = st.radio(
        label=f"label_{item['key']}", 
        options=likert_options, 
        index=None, 
        key=f"grid_radio_{item['key']}_run_{st.session_state.current_run}", 
        horizontal=True, 
        label_visibility="collapsed"
    )
    
    if choice is not None:
        current_survey_answers[dict_key] = likert_options.index(choice) + 1
    else:
        current_survey_answers[dict_key] = None
        
    st.markdown("<hr style='margin: 8px 0px; border-top: 1px dashed #ccc;'>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# --- DYNAMIC BUTTON TEXT LOGIC ---
# Update the UI button based on whether the participant is continuing or finishing
current_block = st.session_state.current_run
total_blocks = len(st.session_state.block_order)

if current_block < total_blocks:
    next_block = current_block + 1
    button_text = f"Submit Survey & Proceed to Block {next_block} →"
else:
    button_text = "Submit Survey & Proceed to Final Step →"

if st.button(button_text, use_container_width=True):
    # Form Validation: Ensure all 7 Likert items have an active selection before advancing
    if None in current_survey_answers.values():
        st.error("⚠️ Please answer all items before proceeding to the next part.")
    else:
        # Pass the dictionary directly into the callback function
        submit_pipeline_survey(current_survey_answers)
