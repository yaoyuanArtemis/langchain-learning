import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
load_dotenv(override=True)

DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 创建模型部分
model = init_chat_model(model="deepseek-chat",model_provider="deepseek")
system_prompt = ChatPromptTemplate.from_messages([
    ("system","你叫小智，是一名AI助手"),("human","{input}")])
qa_chain = system_prompt | model | StrOutputParser()
res = qa_chain.invoke({"input":"你是谁"})
print(res)
# response函数
async def chat_response(message,history):
    partial_message = ""
    async for chunk in qa_chain.astream({"input":message}):
        partial_message += chunk
        yield partial_message

# 创建前端页面
def crate_frontend():
    css = """
    .main-container{
        max-width:1200px;
        margin: 0 auto;
        padding: 20px
    }
    .header-text{
        text-align : center;
        margin-bottom : 20px
    }
    """
    with gr.Blocks(title="DeepSeek Chat",css=css) as demo:
        with gr.Column(elem_classes=["main-container"]):
            # 居中显示标题
            gr.Markdown(
                "# 🤖高瑜：",
                elem_classes=["header-text"]
            )
            gr.Markdown(
                "流式对话机器人",
                elem_classes=["header-text"]
            )

            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True,
                # avatar_images=(
                #     ""
                # )
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="请输入你的问题",
                    container=False,
                    scale=7
                )
                submit = gr.Button(
                    "发送",
                    scale=1,
                    variant="primary"
                )
                clear = gr.Button(
                    "清空",
                    scale=1
                )
        async def respond(message,chat_history):
            if not message.strip():
                yield "",chat_history
                return
            chat_history = chat_history + [(message,None)]
            # chat_history.append((message,None))
            yield "",chat_history

            async for response_chunk in chat_response(message,chat_history):
                chat_history[-1] = (message,response_chunk) 
                yield "",chat_history
        
        def clear_history():
            return [],""
        
        # msg.submit(respond,[msg,chatbot],[msg,chatbot])
        submit.click(respond,[msg,chatbot],[msg,chatbot])
        clear.click(clear_history,outputs=[chatbot,msg])
    return demo


demo = crate_frontend()
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    debug=True
)