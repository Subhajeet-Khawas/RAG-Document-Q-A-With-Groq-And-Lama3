# RAG Document Q&A With Groq And Llama3

This project is a **Retrieval-Augmented Generation (RAG) system** that allows users to upload PDF documents and ask queries based on the content. The system leverages **Groq's Llama 3 model** and **FAISS-based vector search** for efficient retrieval and question-answering.

## Web app URL:
https://rag-document-q-a-with-groq-and-lama3.streamlit.app


## Features
- **PDF Upload**: Users can upload multiple PDF documents.
- **Document Processing**: Extracts text and splits it into manageable chunks.
- **FAISS Vector Storage**: Creates embeddings using `all-MiniLM-L6-v2` and stores them for retrieval.
- **RAG-based Q&A**: Uses **Groq's Llama 3 model** to answer user queries based on document context.
- **Streamlit UI**: Provides an interactive web-based interface for users.

## Tech Stack
- **Python**
- **Streamlit** (for UI)
- **LangChain** (for RAG pipeline)
- **FAISS** (for vector storage)
- **HuggingFace Embeddings** (`all-MiniLM-L6-v2` for text vectorization)
- **Groq API** (for Llama 3 model)

## Setup and Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/rag-document-qa.git
   cd rag-document-qa
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Set up API keys in a `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```

5. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Upload one or more PDF files.
2. Click **Process PDFs** to generate vector embeddings.
3. Enter a query related to the uploaded documents.
4. View AI-generated answers and document similarity search results.

## Project Structure
```
├── app.py                   # Main Streamlit application
├── requirements.txt         # Required dependencies
├── .env.example             # Template for environment variables
├── README.md                # Project documentation
├── utils/                   # Helper functions
└── models/                  # Pre-trained models and embeddings
```

## Future Improvements
- Support for additional document formats (e.g., TXT, DOCX).
- Improved UI/UX for better user interaction.
- Multi-modal retrieval (text + images from PDFs).



<img width="1435" alt="image" src="https://github.com/user-attachments/assets/ebb1154e-0d46-467e-b6f7-256476e1c8a3" />
