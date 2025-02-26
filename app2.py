
import os
import getpass
from dotenv import load_dotenv
import bs4
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Define configuration and helper classes
class Config:
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not set. Check your .env file.")

        if not os.environ.get("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

        self.llm_model = "llama3-8b-8192"
        self.embeddings_model = "llama3.2"

# Define the loader and document processing classes
class DocumentLoader:
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.loader = UnstructuredURLLoader(urls=urls)

    def load_documents(self):
        print('Loading documents...')
        docs = self.loader.load()
        return docs

class DocumentSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, docs):
        print('Splitting documents...')
        return self.text_splitter.split_documents(docs)

class EmbeddingProcessor:
    def __init__(self, embeddings_model: str):
        self.embeddings = OllamaEmbeddings(model=embeddings_model)

    def filter_zero_vectors(self, docs):
        non_zero_docs = []
        for doc in docs:
            embedding = self.embeddings.embed_documents(doc.page_content)
            if any(embedding):  # Checks if there's at least one non-zero value in the vector
                non_zero_docs.append(doc)
            else:
                print(f"Skipping document with zero vector: {doc.page_content[:100]}")
        return non_zero_docs

# Define the vector store manager class
class VectorStoreManager:
    def __init__(self, pinecone_api_key: str, index_name: str):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        if self.index_name not in [i.name for i in self.pc.list_indexes()]:
            self._create_index()

        self.index = self.pc.Index(self.index_name)
        self.vector_store = PineconeVectorStore(embedding=OllamaEmbeddings(model="llama3.2"), index=self.index)

    def _create_index(self):
        print('Creating Pinecone index...')
        self.pc.create_index(
            name=self.index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    def add_documents(self, documents):
        print('Adding documents to the vector store...')
        self.vector_store.add_documents(documents=documents)

    def similarity_search(self, query):
        print(f'Retrieving documents for query: {query}')
        return self.vector_store.similarity_search(query)

# Define the application state and flow
class State(TypedDict):
    question: str
    context: List
    answer: str

class QASystem:
    def __init__(self, config: Config, vector_store_manager: VectorStoreManager, embeddings_processor: EmbeddingProcessor):
        self.config = config
        self.vector_store_manager = vector_store_manager
        self.embeddings_processor = embeddings_processor

        # Initialize the language model
        self.llm = init_chat_model(self.config.llm_model, model_provider="groq")

        # Set up the prompt template
        self.prompt_template = """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question for the webpage article. 
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. 
            Question: {question}
            Context: {context}
            Answer:
        """
        self.prompt = PromptTemplate(input_variables=["question", "context"], template=self.prompt_template)

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store_manager.similarity_search(state["question"])
        print({"context": retrieved_docs})
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        print({"answer": response.content})
        return {"answer": response.content}

    def process_question(self, question: str):
        state = State(question=question, context=[], answer="")
        
        # Retrieve context for the question
        retrieve_result = self.retrieve(state)
        state["context"] = retrieve_result["context"]

        # Generate an answer based on the retrieved context
        generate_result = self.generate(state)
        state["answer"] = generate_result["answer"]
        return state["answer"]

# Main execution
if __name__ == '__main__':
    config = Config()
    document_loader = DocumentLoader(urls=["https://lilianweng.github.io/posts/2023-06-23-agent/"])
    docs = document_loader.load_documents()

    document_splitter = DocumentSplitter()
    all_splits = document_splitter.split_documents(docs)

    embeddings_processor = EmbeddingProcessor(config.embeddings_model)
    all_splits_filtered = embeddings_processor.filter_zero_vectors(all_splits)

    vector_store_manager = VectorStoreManager(pinecone_api_key=config.pinecone_api_key, index_name=config.index_name)
    vector_store_manager.add_documents(all_splits_filtered)

    qa_system = QASystem(config, vector_store_manager, embeddings_processor)

    # Interactive question-answering loop
    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        
        answer = qa_system.process_question(question)
        print("\nAnswer:", answer)