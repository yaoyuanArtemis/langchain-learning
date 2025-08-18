from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_experimental.tools import PythonAstREPLTool
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv(override=True)

# 加载模型API_KEY
DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
model = init_chat_model(model="deepseek-chat",model_provider="deepseek")
df = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# pd.set_option("max_colwidth",200)
# print(dataset.head(5))


# 使用Python解释器
tool = PythonAstREPLTool(locals={"df":df})
# res = tool.invoke("df['SeniorCitizen'].mean()")
# print(res)
llm_with_tools = model.bind_tools([tool])
# res = llm_with_tools.invoke(
#     '我有一张表，名为df，请帮我计算SeniorCitizen字段的均值'
# )
# print(res)


# 结果解析器提取Python代码
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
parser = JsonOutputKeyToolsParser(key_name=tool.name,first_tool_only=True)
llm_chain = llm_with_tools | parser

system = """
 你可以访问一个名为pandas的数据框,你可以使用df.head().to_markdown()查看数据集的基本信息,请根据用户提出的问题,编写Python代码来回答。只返回代码，不返回其他内容。只允许使用pandas和内置库
"""

prompt = ChatPromptTemplate([
    ('system',system),
    ('user',"{question}")
])

full_chain = prompt | llm_with_tools | parser | tool
res = full_chain.invoke({"question":"请帮我计算SeniorCitizen字段的均值"})
# print(res)
res2 = full_chain.invoke({"question":"请帮我分析gender,SeniorCitizen字段之间的相关关系"})
print(res2)


# 增加打印结果节点
from langchain_core.runnables import RunnableLambda
def code_print(res):
    print("即将运行Python代码",res['query'])
    return res

print_node = RunnableLambda(code_print)
full_chain_with_print = prompt | llm_with_tools | parser | print_node | tool 
full_chain_with_print.invoke({"question":"请帮我分析gender,SeniorCitizen字段之间的相关关系"})