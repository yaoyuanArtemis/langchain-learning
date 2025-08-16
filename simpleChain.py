# 一个经典的chain由三部分构成：提示词模版 + 模型 + 输出解析器

from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# 加载模型API_KEY
DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
model = init_chat_model(model="deepseek-chat",model_provider="deepseek")



# base chain
# prompt_template = ChatPromptTemplate([("system","你是一个乐于助人的助手"),("user","这是用户的问题：{topic},请用yes或no来回答")])
# bool_qa_chain = prompt_template | model | BooleanOutputParser()

# question = "请问1+1是否大于2?"
# result = bool_qa_chain.invoke(question)
# print(result)

# base Chain with schemas
from langchain.output_parsers import ResponseSchema,StructuredOutputParser
# schemas = [
#     ResponseSchema(name="age",description="用户的年龄"),
#     ResponseSchema(name="name",description="用户名称")
# ]

# parser = StructuredOutputParser.from_response_schemas(schemas)
# prompt = PromptTemplate.from_template("请根据以下内容提取用户信息，并返回JSON格式:\n{input}\n\m{format_instructions}")
# chain = (prompt.partial(format_instructions=parser.get_format_instructions())  | model | parser)
# result = chain.invoke({"input":"用户叫刘峰,去年30岁"})
# print(result)

# complex chain
news_gen_prompt = PromptTemplate.from_template("请根据以下新闻标题写一份简短的新闻内容(100字以内):标题:{title}")
news_chain = news_gen_prompt | model

schemas = [ResponseSchema(name="time",description="事件发生时间"),ResponseSchema(name="location",description="事件发生的地点"),ResponseSchema(name="event",description="发生的具体事件")]
parser = StructuredOutputParser.from_response_schemas(schemas)
summary_prompt = PromptTemplate.from_template("请从下面新闻内容中提取关键信息,并返回JSON格式:{news}{format_instructions}")
summary_chain = summary_prompt.partial(format_instructions=parser.get_format_instructions()) | model | parser
print(parser.get_format_instructions())
full_chain = news_chain | summary_chain

result = full_chain.invoke({"title":"英伟达公司在加州发布了新的GPU"})
print(result)