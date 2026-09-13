from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Load the local incident knowledge base document
loader = TextLoader("incident_kb.txt", encoding="utf-8")
documents = loader.load()

# 2. Split the document into smaller chunks for precise retrieval
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Initialize local embedding model (runs completely offline)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Store and persist vectors in local Chroma database
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# 5. Connect to local Ollama instance running Llama 3
llm = Ollama(model="llama3")

# 6. Build the retrieval-augmented question answering chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1})
)

# 7. Test query execution
print("=== Local Incident Troubleshooting Assistant Ready ===")
query = "What should I do if I get ERR_ITS_COMM_TIMEOUT?"
print(f"Query: {query}\n")

response = qa_chain.run(query)
print(f"Assistant Response:\n{response}")