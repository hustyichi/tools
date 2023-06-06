import faiss

from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.tools import DuckDuckGoSearchRun
from langchain.vectorstores import FAISS

from app.tools.web_query_tool import WebpageQATool

embeddings_model = OpenAIEmbeddings()
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1.0)
query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))
web_search = DuckDuckGoSearchRun()

tools = [
    web_search,
    query_website_tool,
    # HumanInputRun(), # Activate if you want the permit asking for help from the human
]

agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
    # human_in_the_loop=True, # Set to True if you want to add feedback at each step.
)


if __name__ == "__main__":
    agent.run(["四岁女孩童书推荐"])
