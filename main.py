from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from langchain_community.document_loaders import WebBaseLoader

#Langchain WebBase loader is a simple webloader that works for most of the URLs. We will use some of the excellent langchain blogs for this exercise

urls = ["https://blog.langchain.dev/langgraph-multi-agent-workflows/",
        "https://blog.langchain.dev/langgraph/",
        "https://blog.langchain.dev/deconstructing-rag/"]

docs = [WebBaseLoader(url).load() for url in urls]

def flatten(docs):
    flat_list = []
    for row in docs:
        flat_list += row
    return flat_list

final_docs = flatten(docs)
print(final_docs[0].page_content[:200])

#we will use Recursive Character Text Splitter for this exercise, as it works well for text based contents (remember, our documents are blogs)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(final_docs)

#Now lets store the chunks into a Chroma vector database. We will persist the data for future retrievals.

vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-large"),persist_directory = "mydb")

#Now lets define our retriever. Retriever helps in fetching relevant documents from vector db based on semantic similarity of the user query
retriever = vectorstore.as_retriever()

#define a simple prompt for RAG. This is taken from langchain prompt hub

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

#we will use llama3 with ChatGroq. ChatGroq inference speed is awesome.

model = ChatGroq(temperature=0, model_name="llama3-70b-8192")

chain = (RunnableParallel({"context": retriever,"question": RunnablePassthrough()})
    | prompt
    | model
)

#now lets invoke the chain with the question.
response = chain.invoke("what is routing") 
print(response.content) 