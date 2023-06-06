
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from loguru import logger

from app.tools.web_query_tool import WebpageQATool

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1.0)
query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))

if __name__ == "__main__":
    url = "https://zhuanlan.zhihu.com/p/64506277"
    question = "四岁女孩值的推荐的童书有哪些"
    ret = query_website_tool.run({"url": url, "question": question})
    logger.info("-> final result : {}", ret)
