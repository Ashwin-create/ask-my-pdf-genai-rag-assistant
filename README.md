# ask-my-pdf-genai-rag-assistant
A GenAI RAG assistant designed to comprehend your PDF's context and provide accurate answers to your questions quickly-give it a try!(LangChain-Llama3-Google GenAI-Groq API)


Deployed at: https://ask-my-pdf-genai-rag-assistant-ewdn3vmjfdxngwairpfhcf.streamlit.app/


Check out my latest GenAI build- Ask my PDF: Q&A Retrieval-Augmented Generation (RAG) system, powered by LangChain with Llama3, Google's GenAI Embeddings, and the Groq API.

Why LangChain? 🔗
LangChain is the backbone of this app, enabling seamless integration of 'Llama3-8b-8192', and document loaders. Its modular structure made it easy to build a highly scalable and customizable RAG system. With LangChain, I could handle document ingestion, text chunking, retrieval, and context-based question-answering efficiently, ensuring accurate

Why Llama3? 
Llama3’s ability to handle large contexts (up to 8K tokens) provides depth in understanding lengthy documents. Its performance balance and efficient resource consumption made it the best choice for extracting precise answers from PDFs.

Why Google GenAI Embeddings?
Google's GenAI Embeddings service excels at creating dense vector representations, ensuring that context and meaning from the PDF are preserved in the vector store(knowledge base) for precise question-answering. This embedding model works seamlessly with LangChain, enhancing the app's retrieval capabilities.

Why Groq API?
Groq API's high-speed performance and reliability allow me to leverage large-scale AI models efficiently. Its streamlined integration with LangChain ensures smooth model deployment without compromising response times. (0.11 seconds for a 40 page pdf - not bad!)

