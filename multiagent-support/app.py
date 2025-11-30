import streamlit as st
from chat import MultiAgentChat
from langchain_core.messages import AIMessage, HumanMessage

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Customer Support",
    page_icon="",
    layout="wide"
)

# Load API key from secrets
API_KEY = st.secrets["anthropic_api_key"]

# Initialize session state
if "support_tickets" not in st.session_state:
    st.session_state.support_tickets = {}

if "chat_system" not in st.session_state:
    st.session_state.chat_system = MultiAgentChat(API_KEY, st.session_state.support_tickets)

if "history" not in st.session_state:
    st.session_state.history = []

if "customer_name" not in st.session_state:
    st.session_state.customer_name = ""

# === MAIN UI ===
st.title("Multi-Agent Customer Support Simulation")
st.markdown("---")

# === SIDEBAR: Database View ===
with st.sidebar:
    st.header("📊 Support Tickets Database")
    st.markdown("---")
    
    if st.session_state.support_tickets:
        for ticket_num, ticket_info in st.session_state.support_tickets.items():
            with st.expander(f"Ticket #{ticket_num}"):
                st.write(f"**Status:** {ticket_info['status']}")
                st.write(f"**Type:** {ticket_info['type']}")
                st.write(f"**Customer:** {ticket_info.get('customer_name', 'Unknown')}")
                st.write(f"**Message:** {ticket_info['message'][:100]}...")
                
                # Add ability to update ticket status
                new_status = st.selectbox(
                    "Update Status:",
                    ["unresolved", "in_progress", "resolved"],
                    index=["unresolved", "in_progress", "resolved"].index(ticket_info['status']),
                    key=f"status_{ticket_num}"
                )
                if st.button("Update", key=f"update_{ticket_num}"):
                    st.session_state.support_tickets[ticket_num]['status'] = new_status
                    st.success(f"Ticket #{ticket_num} updated!")
                    st.rerun()
    else:
        st.info("No tickets yet")
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.history = []
        st.rerun()
    
    # Clear all tickets button
    if st.button("🗑️ Clear All Tickets"):
        st.session_state.support_tickets = {}
        st.rerun()

# === MAIN CONTENT ===
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("ℹ️ How to Use")
    st.markdown("""
    **Types of Messages:**
    
    ✅ **Positive Feedback**
    - "Thank you for your help!"
    - "Great service!"
    
    ❌ **Negative Feedback**
    - "This is not working"
    - "I'm very disappointed"
    
    🔍 **Query**
    - "What's the status of ticket #123456?"
    - "Check ticket 654321"
    """)

with col1:
    # Customer name input
    if not st.session_state.customer_name:
        st.subheader("👤 Welcome!")
        customer_name_input = st.text_input("Please enter your name to start:", key="name_input")
        if st.button("Start Chat"):
            if customer_name_input:
                st.session_state.customer_name = customer_name_input
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")
    else:
        st.subheader(f"👤 Hello, {st.session_state.customer_name}!")
        
        # Chat interface
        st.markdown("### 💬 Conversation")
        
        # Display conversation history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message['content'])
                else:
                    with st.chat_message("assistant"):
                        st.write(message['content'])
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to history
            st.session_state.history.append({"role": "user", "content": user_input})
            
            # Process message through multi-agent system
            with st.spinner("Processing..."):
                result = st.session_state.chat_system.process_message(
                    st.session_state.history,
                    st.session_state.customer_name
                )
                
                # Extract assistant response - handle both dict and AIMessage
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    assistant_message = last_message.content
                elif isinstance(last_message, dict):
                    assistant_message = last_message["content"]
                else:
                    assistant_message = str(last_message)
                
                st.session_state.history.append({"role": "assistant", "content": assistant_message})
            
            # Rerun to display new messages
            st.rerun()

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Multi-Agent Customer Support System | Powered by Claude & LangGraph</p>
</div>
""", unsafe_allow_html=True)
