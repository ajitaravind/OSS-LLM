from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from langchain_community.document_loaders import WebBaseLoader

#Langchain WebBase loader is a simple webloader that works for most of the URLs.

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

#print first 200 characters to check how the documents look, this is a quick sanity check

print(docs[0].page_content[:200])

#we will use Recursive Character Text Splitter for this exercise, as it works well for text based contents (remember, our document is a blog)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(docs)

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

#now lets define our chain using Langchain Expression language. Simple way to understand this is, remember our prompt requires a dictionary
#with two input variables - context and question (check our prompt again). We need to provide them right?
#context - its a list of documents fetched by our retriever, we invoke retriever for this purpose.
#question - we will just pass it along without doing any manipulation, thats why we are using RunnablePassthrough() runnable

chain = (RunnableParallel({"context": retriever,"question": RunnablePassthrough()})
    | prompt
    | model
)

#now lets invoke the chain with the question.
response = chain.invoke("what is self reflection") 

print(response.content) #Self-reflection is a vital aspect that allows autonomous agents to improve iteratively by refining past action decisions and correcting previous mistakes. It plays a crucial role in real-world tasks.


