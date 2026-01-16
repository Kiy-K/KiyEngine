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
mcp_server_url = os.getenv("MCP_SERVER_URL")

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
    st.info(f"🔗 MCP Server: `{mcp_server_url or 'Not configured'}`")
    
    # Show available MCP tools
    if st.button("🔍 List MCP Tools"):
        with st.spinner("Fetching tools..."):
            try:
                mcp_tools = tools.list_mcp_tools()
                st.success("Available Tools:")
                for tool in mcp_tools:
                    st.code(str(tool))
            except Exception as e:
                st.error(f"Failed to list tools: {e}")
    
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.display_messages = []
        st.rerun()

# 3. Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

st.title("MedGemma Triage 🏥")
st.caption("Powered by ReAct + FastMCP Medical Tools")


def _render_assistant_message(parsed: dict):
    """Helper to render parsed assistant message."""
    if parsed.get("thought"):
        with st.expander("🧠 AI Thinking Process"):
            st.write(parsed["thought"])
    
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
        st.markdown(parsed.get("content", "No content."))


# 4. Display History
for msg in st.session_state.display_messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            _render_assistant_message(msg["parsed"])


# 5. Chat Input & ReAct Loop
if user_input := st.chat_input("Describe patient symptoms..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)

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
                
                # Check for [SEARCH: ...] or [TOOL: tool_name, args]
                search_match = re.search(r'\[SEARCH:\s*(.*?)\]', raw_content)
                tool_match = re.search(r'\[TOOL:\s*(\w+),\s*(\{.*?\})\]', raw_content, re.DOTALL)
                
                if search_match:
                    query = search_match.group(1).strip()
                    
                    with st.status(f"🔍 Calling MCP Tool for: '{query}'...", expanded=True) as status:
                        # Use MCP server for search
                        search_result = tools.call_mcp_tool("search", {"query": query})
                        st.write(search_result[:500] + "..." if len(search_result) > 500 else search_result)
                        status.update(label="✅ MCP Tool Complete", state="complete", expanded=False)
                    
                    st.session_state.messages.append({"role": "assistant", "content": raw_content})
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"SYSTEM: MCP Search Results:\n{search_result}\nNow provide your final triage assessment in JSON format."
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
                
                elif tool_match:
                    # Generic tool call: [TOOL: tool_name, {args}]
                    tool_name = tool_match.group(1)
                    try:
                        import json
                        tool_args = json.loads(tool_match.group(2))
                    except:
                        tool_args = {}
                    
                    with st.status(f"🔧 Calling MCP Tool: {tool_name}...", expanded=True) as status:
                        tool_result = tools.call_mcp_tool(tool_name, tool_args)
                        st.write(tool_result[:500] + "..." if len(tool_result) > 500 else tool_result)
                        status.update(label="✅ MCP Tool Complete", state="complete", expanded=False)
                    
                    st.session_state.messages.append({"role": "assistant", "content": raw_content})
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"SYSTEM: Tool '{tool_name}' Results:\n{tool_result}\nNow provide your final triage assessment in JSON format."
                    })
                    
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
                _render_assistant_message(parsed)
                
                st.session_state.messages.append({"role": "assistant", "content": raw_content})
                st.session_state.display_messages.append({"role": "assistant", "parsed": parsed})

            except Exception as e:
                st.error(f"An error occurred: {e}")
