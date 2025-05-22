from importlib import import_module
try:
    conf = import_module("src.configs.real_config")
except ImportError:
    conf = import_module("src.configs.config")

if conf.GPT_api_key != '':
    from .api_model import GPT
elif conf.GPT_transit_url != '':
    from .remote_model import GPT

from .local_model import Qwen_7B_Chat, Baichuan2_13B_Chat, ChatGLM3_6B_Chat, DeepSeek_R1_Distill_Qwen_7B_Chat, Qwen2_5_7B_Chat, Local_Chat
from .remote_model import Qwen_14B_Chat