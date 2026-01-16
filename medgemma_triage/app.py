import streamlit as st
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import utils
import prompts
import tools

# 1. Setup & Config
load_dotenv()

st.set_page_config(
    page_title="MedGemma Triage 🏥",
    page_icon="🏥",
    layout="wide"
)

# Initialize OpenAI Client
hf_endpoint_url = os.getenv("HF_ENDPOINT_URL")
hf_token = os.getenv("HF_TOKEN")

if not hf_endpoint_url or not hf_token:
    st.error("Missing Environment Variables! Please check .env file.")
    st.stop()

client = OpenAI(
    base_url=hf_endpoint_url,
    api_key=hf_token
)

# 2. Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.info("ReAct Mode: AI can search PubMed for evidence.")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.display_messages = []
        st.rerun()

# 3. Session State
if "messages" not in st.session_state:
    st.session_state.messages = []  # Full history (for API)
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # Rendered display

st.title("MedGemma Triage 🏥")
st.caption("Powered by ReAct + PubMed Search")

# 4. Display History
for msg in st.session_state.display_messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            _render_assistant_message(msg["content"]) if callable else None


def _render_assistant_message(parsed: dict):
    """Helper to render parsed assistant message."""
    # Thought Expander
    if parsed.get("thought"):
        with st.expander("🧠 AI Thinking Process"):
            st.write(parsed["thought"])
    
    # Content
    if parsed.get("is_json") and isinstance(parsed.get("content"), dict):
        data = parsed["content"]
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
        # Plain text fallback
        st.markdown(parsed.get("content", "No content."))


# Re-render history
for msg in st.session_state.display_messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            _render_assistant_message(msg["parsed"])


# 5. Chat Input & ReAct Loop
if user_input := st.chat_input("Describe patient symptoms..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)

    # --- ReAct Loop ---
    with st.chat_message("assistant"):
        with st.spinner("Analyzing symptoms..."):
            try:
                # First API Call
                api_messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT}] + \
                               [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                
                response = client.chat.completions.create(
                    model="tgi",
                    messages=api_messages,
                    max_tokens=600,
                    temperature=0.2,
                )
                raw_content = response.choices[0].message.content
                
                # Check for [SEARCH: ...]
                search_match = re.search(r'\[SEARCH:\s*(.*?)\]', raw_content)
                
                if search_match:
                    query = search_match.group(1).strip()
                    
                    # Show search status
                    with st.status(f"🔍 Searching PubMed for: '{query}'...", expanded=True) as status:
                        search_result = tools.search_pubmed(query)
                        st.write(search_result[:500] + "..." if len(search_result) > 500 else search_result)
                        status.update(label="✅ Search Complete", state="complete", expanded=False)
                    
                    # Append to history
                    st.session_state.messages.append({"role": "assistant", "content": raw_content})
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"SYSTEM: Search Results:\n{search_result}\nNow provide your final triage assessment in JSON format."
                    })
                    
                    # Second API Call
                    api_messages_2 = [{"role": "system", "content": prompts.SYSTEM_PROMPT}] + \
                                     [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    
                    response_2 = client.chat.completions.create(
                        model="tgi",
                        messages=api_messages_2,
                        max_tokens=600,
                        temperature=0.2,
                    )
                    raw_content = response_2.choices[0].message.content
                
                # Parse final response
                parsed = utils.parse_medgemma_response(raw_content)
                
                # Render
                _render_assistant_message(parsed)
                
                # Save to display state
                st.session_state.messages.append({"role": "assistant", "content": raw_content})
                st.session_state.display_messages.append({"role": "assistant", "parsed": parsed})

            except Exception as e:
                st.error(f"An error occurred: {e}")
