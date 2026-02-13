import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DB_PATH = os.getenv('CHATBOT_DB_PATH', './data/chatbot.db')

# Ensure data directory exists
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# LLM Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
LLM_MODEL = 'gemini-2.5-flash'
