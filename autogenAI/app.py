import os
import autogen

from dotenv import load_dotenv

load_dotenv()

config_list = [
    {
        'model':'gpt-3.5-turbo-16k',
        'api_key': os.getenv('OPENAI_API_KEY ')
    }
]

llm_config = {
    # 'request_timeout'=600,
    'timeout':600,
    'seed':42,
    'config_list':config_list,
    'temperature':0
}

assistant = autogen.AssistantAgent(
    name = "assistant",
    llm_config = llm_config
)

user_proxy = autogen.UserProxyAgent(
    name = "user_proxy",
    human_input_mode='TERMINATE',
    max_consecutive_auto_reply=10,
    is_termination_msg= lambda x: x.get('content','').rstrip().endswith('TERMINATE'),
    code_execution_config={'work_dir':'web'},
    llm_config=llm_config,
    system_message="""Apply TERMINATE if the task has been solved at full satisfaction.
    Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

task = """
Write a python code to get the prime numbers between 1 to 100.
"""

user_proxy.initiate_chat(
    assistant,
    message=task
)