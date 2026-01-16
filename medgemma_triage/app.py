import streamlit as st
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import utils
import prompts

# 1. Setup & Config
load_dotenv()

# Set page config
st.set_page_config(
    page_title="MedGemma Triage 🏥",
    page_icon="🏥",
    layout="wide"
)

# Initialize OpenAI Client (compatible with HF Endpoints)
hf_endpoint_url = os.getenv("HF_ENDPOINT_URL")
hf_token = os.getenv("HF_TOKEN")

if not hf_endpoint_url or not hf_token:
    st.error("Missing Environment Variables! Please check .env file.")
    st.stop()

# Initialize Client
client = OpenAI(
    base_url=hf_endpoint_url,
    api_key=hf_token
)

# 2. Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.info("Ensure your Hugging Face Endpoint is running.")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 3. Chat Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("MedGemma Triage 🏥")

# Display History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            content = msg["content"]
            
            # Helper to display content
            # If it's a dict (parsed success)
            if isinstance(content, dict) and "data" in content:
                # Thought Process
                if content.get("thought"):
                    with st.expander("🧠 AI Thinking Process"):
                        st.write(content["thought"])
                
                # JSON Data
                data = content["data"]
                triage_level = data.get("triage_level", "Unknown")
                rationale = data.get("clinical_rationale", "No rationale provided.")
                actions = data.get("recommended_actions", [])

                if triage_level == "Emergency":
                    st.error(f"🚨 **TRIAGE LEVEL: {triage_level}**")
                elif triage_level == "Urgent":
                    st.warning(f"⚠️ **TRIAGE LEVEL: {triage_level}**")
                else: # Non-Urgent
                    st.success(f"✅ **TRIAGE LEVEL: {triage_level}**")
                
                st.markdown(f"**Clinical Rationale:**\n{rationale}")
                
                if actions:
                    st.markdown("**Recommended Actions:**")
                    for action in actions:
                        st.markdown(f"- {action}")
            else:
                # Fallback / Error
                st.warning("⚠️ Could not parse structured response.")
                st.write(content)

# 4. Input Handling
if prompt := st.chat_input("Describe patient symptoms..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # API Call
    with st.chat_message("assistant"):
        with st.spinner("Analyzing symptoms..."):
            try:
                response = client.chat.completions.create(
                    model="tgi",
                    messages=[
                        {"role": "system", "content": prompts.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.2,
                )
                
                raw_content = response.choices[0].message.content
                
                # Parse
                parsed_result = utils.parse_medgemma_response(raw_content)
                
                # Display Current Response
                if isinstance(parsed_result, dict) and "data" in parsed_result:
                    if parsed_result.get("thought"):
                        with st.expander("🧠 AI Thinking Process"):
                            st.write(parsed_result["thought"])
                    
                    data = parsed_result["data"]
                    triage_level = data.get("triage_level", "Unknown")
                    rationale = data.get("clinical_rationale", "No rationale provided.")
                    actions = data.get("recommended_actions", [])

                    if triage_level == "Emergency":
                        st.error(f"🚨 **TRIAGE LEVEL: {triage_level}**")
                    elif triage_level == "Urgent":
                        st.warning(f"⚠️ **TRIAGE LEVEL: {triage_level}**")
                    else:
                        st.success(f"✅ **TRIAGE LEVEL: {triage_level}**")
                    
                    st.markdown(f"**Clinical Rationale:**\n{rationale}")
                    if actions:
                        st.markdown("**Recommended Actions:**")
                        for action in actions:
                            st.markdown(f"- {action}")
                else:
                    st.warning("⚠️ Raw Response (Parsing Failed)")
                    st.write(parsed_result)

                # Append to history
                st.session_state.messages.append({"role": "assistant", "content": parsed_result})

            except Exception as e:
                st.error(f"An error occurred: {e}")
