import streamlit as st
from chat_agent import chatbot, db_manager
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ************************ utility functions ************************

def generate_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    db_manager.create_thread(thread_id, title="New Chat")
    st.session_state['message_history'] = []

def load_conversation(thread_id):
    """Load conversation messages from database"""
    messages = db_manager.get_messages(thread_id)
    temp_messages = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = 'user'
        else:
            role = 'assistant'
        temp_messages.append({'role': role, 'content': msg.content})
    
    return temp_messages

# ************************ Session Setup ************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    db_manager.create_thread(thread_id, title="New Chat")

# ************************ Sidebar UI ************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

# Load all threads from database
threads = db_manager.get_threads()

if threads:
    # Display first conversation
    first_thread = threads[0]
    first_thread_id = first_thread['thread_id']
    first_title = first_thread['title'] or "First Chat"
    
    if st.sidebar.button(f"📌 {first_title}", key=f"first_{first_thread_id}"):
        st.session_state['thread_id'] = first_thread_id
        st.session_state['message_history'] = load_conversation(first_thread_id)
    
    # Display other conversations
    if len(threads) > 1:
        st.sidebar.divider()
        for thread in threads[1:]:
            thread_id = thread['thread_id']
            title = thread['title'] or str(thread_id)[:8]
            
            if st.sidebar.button(title, key=thread_id):
                st.session_state['thread_id'] = thread_id
                st.session_state['message_history'] = load_conversation(thread_id)
else:
    st.sidebar.info("No conversations yet. Click 'New Chat' to start!")

# ************************ Main UI ************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # Add user message to in-memory history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    
    # Set thread title to first message if it's still default
    thread_id = st.session_state['thread_id']
    threads = db_manager.get_threads()
    current_thread = next((t for t in threads if t['thread_id'] == thread_id), None)
    
    if current_thread and current_thread['title'] == "New Chat":
        # Use first 50 chars of user message as title
        title = user_input[:50] if len(user_input) <= 50 else user_input[:47] + "..."
        db_manager.update_thread_title(thread_id, title)
    
    with st.chat_message('user'):
        st.text(user_input)

    # Build message objects from history for LangGraph
    message_objects = []
    for msg in st.session_state['message_history']:
        if msg['role'] == 'user':
            message_objects.append(HumanMessage(content=msg['content']))
        else:
            message_objects.append(AIMessage(content=msg['content']))

    # Get AI response with streaming
    with st.chat_message("assistant"):
        def ai_only_stream():
            for state in chatbot.stream({"messages": message_objects}, stream_mode="values"):
                # Get the last message from the stream
                if state.get('messages'):
                    last_msg = state['messages'][-1]
                    if isinstance(last_msg, AIMessage):
                        yield last_msg.content

        ai_message = st.write_stream(ai_only_stream())

    # Save both messages to database
    db_manager.save_message(st.session_state['thread_id'], 'user', user_input)
    db_manager.save_message(st.session_state['thread_id'], 'assistant', ai_message)
    
    # Add to message history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})