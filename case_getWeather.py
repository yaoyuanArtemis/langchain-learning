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

def get_weather(loc):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q":loc,
        "appid":os.getenv("WEATHER_API_KEY"),
        "units":"metric",
        "lang":"zh-cn"
    }
    response = requests.get(url=url,params=params)
    data = response.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return data

print(get_weather("Beijing")["name"])
print(get_weather("Beijing")["weather"][0]["description"])
# print(get_weather("Beijing")["args"])