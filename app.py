import warnings
import os
import pickle
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# Suppress FutureWarnings from torch (and other libraries if needed)
warnings.filterwarnings("ignore", category=FutureWarning)


# Suppress all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# If you want to suppress only LangChain-related deprecation warnings:
warnings.filterwarnings("ignore", category=UserWarning, module='langchain')


class VectorStoreManager:
    """Manages loading and retrieving documents from a FAISS vector store."""
    
    def __init__(self, index_file, model_name):
        self.index_file = index_file
        self.model_name = model_name
        self.vectorstore = None

    def load_vectorstore(self):
        """Load the FAISS vector store from a file."""
        with open(self.index_file, "rb") as f:
            self.vectorstore = pickle.load(f)

    def get_retriever(self):
        """Get a retriever from the vector store."""
        if self.vectorstore:
            return self.vectorstore.as_retriever()
        return None


class TextGenerator:
    """Handles the generation of answers using the Gemini API."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.configure_api()

    def configure_api(self):
        """Configure the Gemini API."""
        genai.configure(api_key=self.api_key)

    def generate_answer(self, retrieved_docs, query):
        """Generate a sales-oriented answer based on retrieved documents and the query."""
        docs_text = " ".join([doc.page_content for doc in retrieved_docs])
        prompt_text = (
            f"بصفتك موظف مبيعات محترف، أجب على الاستفسار التالي باستخدام المستندات المقدمة فقط. "
            f"احرص على أن تكون الإجابة مقنعة، مركزة على الفوائد الرئيسية، وموجهة لإقناع العميل المحتمل. "
            f"استخدم لغة قوية وجذابة، وأبرز المزايا التنافسية للحل المقدم. "
            f"أكد على كيف يمكن للمنتج أو الخدمة تلبية احتياجات العميل وتقديم حلول فريدة تساهم في تحسين تجربته. "
            f"اعرض النقاط الإيجابية بطريقة تفاعلية وجذابة تجعل العميل يشعر أن هذا الحل هو الخيار الأمثل.\n\n"
            f"المستندات: {docs_text}\n"
            f"الاستفسار: {query}\n"
            f"الإجابة (بنبرة موظف مبيعات محترف ومقنعة):"
        )
        return self.generate_answer_from_gemini(prompt_text)

    def generate_answer_from_gemini(self, prompt_text):
        """Generate an answer using the Gemini API based on the provided prompt text."""
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt_text)
        return response.text


class SalesAnswerApp:
    """Main application class to process the user query, retrieve relevant docs, and generate an answer."""
    
    def __init__(self, vectorstore_file, model_name, api_key):
        self.vectorstore_manager = VectorStoreManager(vectorstore_file, model_name)
        self.text_generator = TextGenerator(api_key)

    def run(self, query):
        """Main method to process query and generate an answer."""
        # Step 1: Load vectorstore
        self.vectorstore_manager.load_vectorstore()
        
        # Step 2: Retrieve relevant documents based on the query string
        retriever = self.vectorstore_manager.get_retriever()
        if retriever:
            # Use the updated method 'invoke' instead of the deprecated 'get_relevant_documents'
            retrieved_docs = retriever.invoke(query)
            
            # Step 3: Generate and print the answer
            answer = self.text_generator.generate_answer(retrieved_docs, query)
            print("Generated Answer:", answer)
        else:
            print("Error: Could not retrieve documents.")


if __name__ == "__main__":
    # Configuration
    VECTORSTORE_FILE = "internal_FAQ/vectorstore.pkl"
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    API_KEY = 'AIzaSyAJiQoksPi_v5itQ2NYjALsJ4TArqW0YLs'  # Your actual API key

    # Create the SalesAnswerApp instance
    app = SalesAnswerApp(VECTORSTORE_FILE, MODEL_NAME, API_KEY)

    # Loop to continuously get user input until 'exit' is entered
    while True:
        QUERY = input("Enter your query (type 'exit' to quit): ")
        if QUERY.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break
        app.run(QUERY)



