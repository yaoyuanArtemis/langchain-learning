"""
curl -X GET "https://api.openweathermap.org/data/2.5/weather?lat=44.34&lon=10.99&appid=a4ff2d71325ef27f0040c28b6b88d579"

"""
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_experimental.tools import PythonAstREPLTool
import os,requests,json
from dotenv import load_dotenv
import pandas as pd
load_dotenv(override=True)

from langchain_core.tools import tool

@tool
def get_weather(loc_zh):
    """获取指定城市的实时天气,城市名字必须使用英文"""
    url = "https://api.openweathermap.org/data/2.5/weather"
    city_map = {
        "庆阳":"Qingyang",
        "西安":"Xi'an",
        "兰州":"Lanzhou"
    }
    params = {
        "q":city_map[loc_zh],
        "appid":os.getenv("WEATHER_API_KEY"),
        "units":"metric",
        "lang":"zh-cn"
    }
    response = requests.get(url=url,params=params)
    data = response.json()
    # print(json.dumps(data, indent=2, ensure_ascii=False))
    return data



DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
model = init_chat_model(model="deepseek-chat",model_provider="deepseek")

tools = [get_weather]
llm_model_tools = model.bind_tools(tools)

response = llm_model_tools.invoke("你好,庆阳的天气怎么样")

from langchain_core.runnables import RunnableLambda
def wrap_weather_data(data):
    return {"data": json.dumps(data, ensure_ascii=False)}


from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
parser = JsonOutputKeyToolsParser(key_name=get_weather.name,first_tool_only=True)
get_weather_chain = llm_model_tools | parser | get_weather | RunnableLambda(wrap_weather_data)
# res = get_weather_chain.invoke("西安的天气如何")

# print(get_weather("Beijing")["name"])
# print(get_weather("Beijing")["weather"][0]["description"])
# print(get_weather("Beijing")["args"])



# 解析JSON的chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

output_prompt = PromptTemplate.from_template(
    """
    你将收到一段JSON格式的天气数据,请用简洁自然的方式转述给用户
    以下是JSON格式数据:

    ```json
    {data}
    ```

    请将其转述为中文天气描述,如:"北京当前天气晴，气温为23摄氏度,湿度58%，风速2.1米/秒"
    只返回一句话描述，不要其他说明和解释
    """
)
output_chain = output_prompt | model | StrOutputParser()

full_chain = get_weather_chain | output_chain
# response = full_chain.invoke("请问西安天气怎么样")
# print(response)


# 创建agent以及运行AgentExecutor
from langchain.agents import create_tool_calling_agent
tools = [get_weather]
prompt_agent = ChatPromptTemplate.from_messages([
    ("system","你是天气助手，协助用户给出相应天气信息"),
    ("human","{input}"),
    ("placeholder","{agent_scratchpad}")
])
agent = create_tool_calling_agent(model,tools,prompt=prompt_agent)
from langchain.agents import AgentExecutor
agent_exeutor = AgentExecutor(agent=agent,tools=tools,verbose=True)
response = agent_exeutor.invoke({"input":"请问今天西安的天气怎么样"})
print(response)
