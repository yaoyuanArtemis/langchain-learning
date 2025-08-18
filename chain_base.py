import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(override=True)

DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 初始化OpenAI客户端
# client = OpenAI(api_key=DeepSeek_API_KEY,base_url="https://api.deepseek.com")
# print(DeepSeek_API_KEY)

# 调用DeepSeek API生成回答
# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role":"system","content":"你是陕西西安人，以一个导游的身份介绍西安"},
#         {"role":"user","content":"你好，请介绍自己"},
#     ]
# )

# print(response.choices[0].message.content)

# 创建一个模型对象
from langchain.chat_models import init_chat_model

model = init_chat_model(model="deepseek-chat",model_provider="deepseek")
question = "你好，介绍一下你自己"
result = model.invoke(question)
print(result)

