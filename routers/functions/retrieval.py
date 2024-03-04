from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import load_tools, initialize_agent

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from huggingface_hub import snapshot_download

# LLMの定義
model_id = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"
download_path = snapshot_download(repo_id=model_id)

tokenizer = AutoTokenizer.from_pretrained(download_path)
model = AutoModelForCausalLM.from_pretrained(download_path)

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=pipe)

# 入力に質問を受け取り、出力に質問に対する回答を返す関数
def get_answer_with_search(question: str) -> str:

    # 質問のテンプレート
    template = """<s>[INST] <<SYS>>
    あなたは誠実で優秀な日本人のアシスタントです。日本語で回答してください。
    <</SYS>>

    質問：{question} [/INST]"""

    prompt_template = ChatPromptTemplate.from_template(template)
    
    tool_names = ["serpapi"]

    tools = load_tools(tool_names)

    agent = initialize_agent(tools, llm, agent="self-ask-with-search")

    output = agent.run(prompt_template.format(question=question))

    return output
