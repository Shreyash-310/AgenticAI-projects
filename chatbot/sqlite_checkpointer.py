from typing import Optional
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
)
from langchain_core.messages import HumanMessage, AIMessage 
from db_manager import DatabaseManager


class SqliteCheckpointer(BaseCheckpointSaver):
    """Custom SQLite checkpointer for LangGraph"""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self.db = db_manager
    
    def put(
        self,
        config,
        checkpoint,
        metadata,
        checkpoint_id,
    ) -> Optional[str]:
        """Save checkpoint to database"""
        thread_id = config.get("configurable", {}).get("thread_id")
        
        if not thread_id:
            return None
        
        # Create thread if it doesn't exist
        if not self.db.thread_exists(thread_id):
            self.db.create_thread(thread_id)
        
        return checkpoint_id
    
    def get_tuple(self, config) -> Optional[CheckpointTuple]:
        """Retrieve checkpoint from database"""
        thread_id = config.get("configurable", {}).get("thread_id")
        
        if not thread_id:
            return None
        
        # Get messages from database
        messages = self.db.get_messages(thread_id)
        
        if not messages:
            return None
        
        # Create checkpoint with correct structure
        return CheckpointTuple(
            config=config,
            checkpoint={
                "channel_values": {
                    "messages": messages
                },
                "metadata": {},
                "checkpoint_id": "",
                "ts": ""
            },
            metadata={},
        )
    
    def put_writes(
        self,
        config,
        writes,
        base_checkpoint_id,
    ) -> Optional[str]:
        """Save writes/messages to database"""
        thread_id = config.get("configurable", {}).get("thread_id")
        
        if not thread_id:
            return None
        
        # Process writes - typically these are messages added to the state
        for key, values in writes.items():
            if key == "chat_node":
                if isinstance(values, list):
                    for value_dict in values:
                        if isinstance(value_dict, dict) and "messages" in value_dict:
                            for msg in value_dict["messages"]:
                                # Determine role and save
                                role = self._get_message_role(msg)
                                content = self._get_message_content(msg)
                                self.db.save_message(thread_id, role, content)
        
        return base_checkpoint_id
    
    def _get_message_role(self, msg) -> str:
        """Extract role from message object"""
        if isinstance(msg, HumanMessage):
            return "user"
        elif isinstance(msg, AIMessage):
            return "assistant"
        else:
            return "system"
    
    def _get_message_content(self, msg) -> str:
        """Extract content from message object"""
        if hasattr(msg, 'content'):
            return msg.content
        elif isinstance(msg, dict) and 'content' in msg:
            return msg['content']
        return str(msg)
