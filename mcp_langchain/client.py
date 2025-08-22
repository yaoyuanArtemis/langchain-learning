from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain import hub
from langchain.agents import AgentExecutor,create_openai_tools_agent
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv(override=True)

DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")

## 初始化playwright浏览器
sync_playwright = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_playwright)
tools = toolkit.get_tools()

## 拉提示词模版
prompt = hub.pull("hwchase17/openai-tools-agent")

model = init_chat_model("deepseek-chat",model_provider="deepseek")

agent = create_openai_tools_agent(model,tools=tools,prompt=prompt)

agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

if __name__ == "__main__":
    command = {
        "input":"访问这个网站 https://yaoyuanartemis.github.io/ 并总结内容"
    }
    response = agent_executor.invoke(command)
    print(response)