# LangGraph Chatbot
A chatbot application with thread-based conversations and persistent SQLite storage.

## 📁 Project Structure

```
chatbot/
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration and environment setup
│   ├── agent/
│   │   ├── __init__.py
│   │   └── chat_agent.py            # LangGraph chatbot agent
│   └── database/
│       ├── __init__.py
│       ├── db_manager.py            # SQLite database operations
│       └── sqlite_checkpointer.py   # LangGraph checkpointer
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py             # Streamlit web UI
├── data/
│   └── chatbot.db                   # SQLite database (auto-created)
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables
└── README.md
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_api_key_here
CHATBOT_DB_PATH=./data/chatbot.db
```

### 3. Run Console Chatbot
```bash
python -m src.agent.chat_agent
```

### 4. Run Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```

## 📊 Database Schema

### `threads` table
- `thread_id` (TEXT, PRIMARY KEY) - Unique conversation identifier
- `created_at` (TEXT) - Creation timestamp
- `updated_at` (TEXT) - Last update timestamp
- `title` (TEXT) - Conversation title

### `messages` table
- `id` (INTEGER, PRIMARY KEY) - Auto-increment ID
- `thread_id` (TEXT, FOREIGN KEY) - Reference to thread
- `role` (TEXT) - Message role ('user' or 'assistant')
- `content` (TEXT) - Message content
- `created_at` (TEXT) - Creation timestamp

## ✨ Features

✅ **Persistent Conversations** - All messages saved to SQLite
✅ **Multi-threaded Support** - Multiple conversation threads
✅ **Thread Management** - Create, load, and switch between threads
✅ **LangGraph Integration** - State management with checkpointing
✅ **Web UI** - Streamlit interface for interactive chat
✅ **Console Interface** - CLI option for testing

## 🔧 Configuration

Database path can be customized via environment variable:
```env
CHATBOT_DB_PATH=./data/chatbot.db  # Default location
```

## 📝 Usage

### Console Usage
```python
from src.agent.chat_agent import chatbot, db_manager

# Conversations are automatically saved
```

### Streamlit Usage
- Click "New Chat" to start a new conversation
- Select previous conversations from the sidebar
- Messages persist across sessions
