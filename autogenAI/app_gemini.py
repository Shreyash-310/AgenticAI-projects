import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import chromadb
from PIL import Image
from termcolor import colored

import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.img_utils import _to_pil, get_image_data
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.code_utils import DEFAULT_MODEL, UNKNOWN, content_str, execute_code, extract_code, infer_lang

from dotenv import load_dotenv

load_dotenv(dotenv_path='../.env')

config_list_gemini = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gemini-pro", "gemini-1.5-pro"],
    },
)

seed = 25  # for caching

# Gemini Assistant
assistant = AssistantAgent(
    "assistant",
    llm_config={
        "config_list":config_list_gemini,
        "seed":seed
    },
    max_consecutive_auto_reply=3
)
user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config={
        "work_dir":"coding"},
        # "user_docker":False},
    human_input_mode="NEVER",
    is_termination_msg=lambda x: content_str(x.get("content")).find("TERMINATE")>=0,
)

result = user_proxy.initiate_chat(
    assistant,
    message="Sort the array with Bubble Sort: [4, 1, 5, 2, 3]"
)