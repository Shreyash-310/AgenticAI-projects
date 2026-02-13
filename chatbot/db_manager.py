import sqlite3
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from config import DB_PATH

class DatabaseManager:
    """Manages SQLite database operations for chatbot conversations"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def init_db(self):
        """Initialize database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create threads table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT
            )
        ''')
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (thread_id) REFERENCES threads (thread_id)
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def create_thread(self, thread_id: str, title: Optional[str] = None) -> bool:
        """Create a new conversation thread"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO threads (thread_id, created_at, updated_at, title)
                VALUES (?, ?, ?, ?)
            ''', (thread_id, now, now, title))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            # Thread already exists
            return False
    
    def get_threads(self) -> List[dict]:
        """Get all threads ordered by most recent"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT thread_id, created_at, updated_at, title
            FROM threads
            ORDER BY updated_at DESC
        ''')
        
        threads = []
        for row in cursor.fetchall():
            threads.append({
                'thread_id': row[0],
                'created_at': row[1],
                'updated_at': row[2],
                'title': row[3]
            })
        
        conn.close()
        return threads
    
    def update_thread_title(self, thread_id: str, title: str) -> bool:
        """Update thread title"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE threads SET title = ? WHERE thread_id = ?
            ''', (title, thread_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating thread title: {e}")
            return False
    
    def save_message(self, thread_id: str, role: str, content: str) -> bool:
        """Save a message to database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            
            # Save message
            cursor.execute('''
                INSERT INTO messages (thread_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
            ''', (thread_id, role, content, now))
            
            # Update thread's updated_at
            cursor.execute('''
                UPDATE threads SET updated_at = ? WHERE thread_id = ?
            ''', (now, thread_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving message: {e}")
            return False
    
    def get_messages(self, thread_id: str) -> List[BaseMessage]:
        """Get all messages for a thread"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT role, content FROM messages
            WHERE thread_id = ?
            ORDER BY created_at ASC
        ''', (thread_id,))
        
        messages = []
        for row in cursor.fetchall():
            role, content = row
            if role == 'user':
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        
        conn.close()
        return messages
    
    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and all its messages"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM messages WHERE thread_id = ?', (thread_id,))
            cursor.execute('DELETE FROM threads WHERE thread_id = ?', (thread_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error deleting thread: {e}")
            return False
    
    def thread_exists(self, thread_id: str) -> bool:
        """Check if thread exists"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT 1 FROM threads WHERE thread_id = ?', (thread_id,))
        exists = cursor.fetchone() is not None
        
        conn.close()
        return exists
