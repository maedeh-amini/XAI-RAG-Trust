"""
Streamlit Page 7: Final Completion & Compensation.

This is the concluding page of the UniCompanion study. It is only accessible 
after the user has evaluated all 4 pipelines and successfully submitted their 
response payload to the server on the previous page.

It provides a formal study debriefing and a secondary form to collect contact 
information (email and optional SONA ID) for participant compensation 
(VP hours or the voucher raffle). 

To ensure strict data privacy and anonymity, this script saves the contact 
information to a completely separate, unlinked remote database worksheet ("Emails"), 
preventing any cryptographic mapping between a participant's identity and their 
evaluation responses.
"""

import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

# --- STRICT SESSION GUARD ---
# Security check: Ensure the participant actually evaluated questions.
# If they attempt to load this URL directly to claim credit without 
# completing the study, they are routed back to the consent page.
if "active_questions" not in st.session_state or len(st.session_state.active_questions) == 0:
    st.switch_page("pages/1_consent.py")

# Add the visual title to the actual page
st.title("🤖 UniCompanion")

# Study is fully complete
st.progress(1.0)
col_meta1, col_meta2 = st.columns([1, 1])
with col_meta1:
    st.caption(f"🆔 **Participant:** ID-{st.session_state.participant_id}")
st.markdown("---")

def save_submission_to_server(email_address, sona_id=""):
    """
    Saves the participant's compensation details to a separate database table.
    
    This is intentionally decoupled from the `ExperimentData` worksheet updated 
    in page 6 to maintain GDPR compliance and double-blind anonymity.
    
    Args:
        email_address (str): The participant's contact email.
        sona_id (str): Optional tracking ID for Psychology students claiming VP hours.
    """
    conn = st.connection("gsheets", type=GSheetsConnection)
    existing_data = conn.read(worksheet="Emails", ttl=0)

    # Prevent duplicate entries if the user double-clicks submit or refreshes
    if existing_data is not None and "email_address" in existing_data.columns and email_address in existing_data["email_address"].values:
        return
        
    new_df = pd.DataFrame([{"email_address": email_address, "sona_id": sona_id}])
    updated_df = pd.concat([existing_data, new_df], ignore_index=True) if existing_data is not None else new_df
    conn.update(worksheet="Emails", data=updated_df)

if "study_completed" not in st.session_state:
    st.session_state.study_completed = False

# Render the final success screen if the user has submitted their email
if st.session_state.study_completed:
    st.balloons()
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.success("✅ **All data saved successfully.**")
    st.markdown(
        """
        <div style='text-align: center; padding: 20px;'>
            <h2>Experiment Complete</h2>
            <p>Thank you for contributing to this HCI research!</p>
            <p><strong>You may now safely close this browser tab/window.</strong></p>
        </div>
        """, 
        unsafe_allow_html=True
    )
else:
    # Render the Debriefing and Compensation form
    st.success("🎉 Evaluation Session Completed Successfully!")
    
    with st.container(border=True):
        st.subheader("ℹ️ Study Debriefing")
        st.write("Thank you for taking part in this study.")
        st.write("The purpose of this experiment was to examine how varying levels of explainability and different background architectures influence a user's perception of transparency and trustworthiness in educational AI assistants.")
        st.write("By evaluating these distinct system behaviors, your responses will directly contribute to designing more reliable and comprehensible digital tools for university environments.")
        st.write("Your responses remain strictly anonymous and will be used solely for research purposes. If you have any questions regarding the study or your data, you may contact the primary researcher at **mamini@uni-osnabrueck.de**.")
        
    st.markdown("---")
    st.subheader("🎁 Compensation (VP-Stunden / Voucher Raffle)")
    st.write("If you require **VP hours**, want to enter the **100 euro voucher raffle**, or both, please share your details below.")
    st.info("🛡️ **Data Privacy Protection Notice:** To ensure absolute anonymity, your contact details are logged completely independently into a separate, unlinked server storage file. There is no cryptographic reference, participant ID, or one-to-one mapping connecting your identity back to the experimental evaluation values you assigned.")

    # Use st.form to capture the compensation data robustly
    with st.form("vp_email_form", border=True):
        email_input = st.text_input("Enter your email address:", placeholder="username@uni-osnabrueck.de")
        
        is_psychology = st.checkbox("I am a psychology student (requires SONA ID for VP hours)")
        sona_input = st.text_input("Enter your SONA ID:", placeholder="e.g., 123456")
        
        col1, col2 = st.columns(2)
        with col1:
            submit_info = st.form_submit_button("Submit & Finish Study", use_container_width=True)
        with col2:
            skip_email = st.form_submit_button("I don't need credits/raffle (Finish Study)", use_container_width=True)
        
        if submit_info:
            # Basic validation to ensure valid data entry
            if email_input.strip() == "" or "@" not in email_input:
                st.error("Please enter a valid email address.")
            elif is_psychology and sona_input.strip() == "":
                st.error("Psychology students must enter their SONA ID to receive VP hours.")
            else:
                save_submission_to_server(email_input.strip(), sona_id=sona_input.strip() if is_psychology else "")
                st.session_state.study_completed = True
                st.rerun() # Force page reload to show the success screen and balloons
                
        if skip_email:
            st.session_state.study_completed = True
            st.rerun()

