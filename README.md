# Medical Chatbot

An AI-powered medical chatbot that uses **Retrieval-Augmented Generation (RAG)** to answer medical questions using information retrieved from a medical knowledge base.

The project loads medical PDF documents, splits the content into smaller chunks, converts the chunks into vector embeddings, stores them in a FAISS vector database, and retrieves relevant information when a user asks a question. The retrieved information is then provided to a Groq-hosted large language model to generate the final response.

## Features

* PDF-based medical knowledge retrieval
* Retrieval-Augmented Generation (RAG)
* Automatic PDF document loading
* Text chunking using Recursive Character Text Splitter
* Sentence Transformer embeddings
* FAISS vector database
* Groq LLM integration
* Streamlit web interface
* Chat-style conversation history
* Retrieved source document information
* Environment variable support for API keys
* Local vector database storage

## Technologies Used

* Python
* Streamlit
* LangChain
* LangChain Community
* LangChain Classic
* LangChain Hugging Face
* LangChain Groq
* Hugging Face Sentence Transformers
* FAISS
* Groq API
* python-dotenv
* uv

## System Architecture

```text
Medical PDF
     |
     v
PDF Loader
     |
     v
Text Extraction
     |
     v
Text Chunking
     |
     v
Hugging Face Embeddings
     |
     v
FAISS Vector Database
     |
     v
User Question
     |
     v
Similarity Search
     |
     v
Relevant Documents
     |
     v
Groq LLM
     |
     v
Generated Answer
     |
     v
Streamlit Chat Interface
```

## Project Structure

```text
Medical_Chatbot/
│
├── data/
│   └── Medical PDF files
│
├── vectorestore/
│   └── db_faiss/
│       ├── index.faiss
│       └── index.pkl
│
├── .venv/
│
├── .env
├── .gitignore
├── main.py
├── RAG_pipeline.py
├── rag_with_llm.py
├── README.md
└── pyproject.toml
```

## How the RAG Pipeline Works

### 1. Load PDF Documents

The project uses `PyPDFLoader` and `DirectoryLoader` to load PDF files from the `data/` directory.

```python
DATA_DIR = "data/"
```

All PDF files inside the directory are loaded automatically.

### 2. Split the Text into Chunks

The extracted text is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

The chunk overlap helps maintain some context between neighboring chunks.

### 3. Generate Embeddings

The project uses the following Sentence Transformer model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts each text chunk into a numerical vector representation.

### 4. Store Embeddings in FAISS

The generated embeddings are stored in a FAISS vector database.

```text
vectorestore/db_faiss
```

FAISS allows the system to efficiently search for documents that are semantically similar to the user's question.

### 5. Retrieve Relevant Information

When the user submits a question, the system searches the FAISS database and retrieves the three most relevant document chunks.

```python
search_kwargs={"k": 3}
```

### 6. Generate the Answer

The retrieved context is passed to a Groq-hosted LLM.

The current model used by the chatbot is:

```text
openai/gpt-oss-120b
```

The model generates an answer using the retrieved context.

## Streamlit Application

The main user interface is implemented in `main.py` using Streamlit.

The application provides:

* Chat interface
* User questions
* AI-generated answers
* Conversation history
* Retrieved source document information

The application can be started using:

```bash
uv run --active streamlit run main.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jalan538/Medical_Chatbot.git
```

Move into the project directory:

```bash
cd Medical_Chatbot
```

### 2. Create the Virtual Environment

This project uses `uv`.

Install the project dependencies using:

```bash
uv sync
```

### 3. Activate the Virtual Environment

For Git Bash on Windows:

```bash
source .venv/Scripts/activate
```

### 4. Configure the Environment Variables

Create a `.env` file in the project root directory.

```text
GROQ_API_KEY=your_groq_api_key
```

Replace `your_groq_api_key` with your actual Groq API key.

Do not upload the `.env` file to GitHub.

## Preparing the Vector Database

Place your medical PDF files inside:

```text
data/
```

Then run:

```bash
uv run --active python RAG_pipeline.py
```

The pipeline will:

1. Load the PDF files.
2. Extract the text.
3. Split the text into chunks.
4. Generate embeddings.
5. Create the FAISS database.
6. Save the database inside:

```text
vectorestore/db_faiss
```

## Running the RAG Test

The `rag_with_llm.py` file can be used to test the RAG system directly from the terminal.

Run:

```bash
uv run --active python rag_with_llm.py
```

The program will ask:

```text
Write Query Here:
```

Example:

```text
Write Query Here: define cancer
```

The system will retrieve relevant information and generate an answer using the Groq LLM.

## Running the Web Application

To start the Streamlit application:

```bash
uv run --active streamlit run main.py
```

Then open:

```text
http://localhost:8501
```

Example questions:

```text
What is cancer?
What are the symptoms of diabetes?
What is hypertension?
What causes anemia?
```

## Environment Variables

The project uses a `.env` file to store sensitive configuration.

Example:

```text
GROQ_API_KEY=your_groq_api_key
```

The application loads the API key using:

```python
load_dotenv(find_dotenv())
```

The API key is then accessed using:

```python
os.environ.get("GROQ_API_KEY")
```

## Important Security Notes

The following files and directories should not be uploaded to GitHub:

```text
.env
.venv/
data/
vectorestore/
```

The `.gitignore` file is configured to exclude these files.

Never hard-code your Groq API key directly inside Python source code.

## Current Medical Knowledge Base

The current project uses a medical encyclopedia PDF as the knowledge source.

The RAG pipeline currently processes:

```text
The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND (1).pdf
```

The document contains multiple medical topics that are used as the retrieval knowledge base.

The current pipeline has processed:

```text
PDF pages: 759
Text chunks: 7079
```

## Main Files

### `RAG_pipeline.py`

Responsible for creating the knowledge base.

It performs:

```text
PDF Loading
     ↓
Text Chunking
     ↓
Embedding Generation
     ↓
FAISS Database Creation
```

### `rag_with_llm.py`

Responsible for testing the complete RAG pipeline from the terminal.

It performs:

```text
User Question
     ↓
FAISS Retrieval
     ↓
Relevant Context
     ↓
Groq LLM
     ↓
Answer
```

### `main.py`

Provides the Streamlit web interface for interacting with the chatbot.

It provides:

```text
User
 ↓
Streamlit Chat Interface
 ↓
FAISS Retrieval
 ↓
Groq LLM
 ↓
Medical Answer
```

## Example Workflow

```text
1. Add medical PDF
        ↓
2. Run RAG_pipeline.py
        ↓
3. Create FAISS vector database
        ↓
4. Start Streamlit
        ↓
5. User enters medical question
        ↓
6. FAISS searches relevant chunks
        ↓
7. Retrieved context is sent to Groq
        ↓
8. Groq generates the answer
        ↓
9. Answer displayed in Streamlit
```

## Git and GitHub

The project repository is available on GitHub:

```text
https://github.com/jalan538/Medical_Chatbot
```

To check the current Git status:

```bash
git status
```

To add changes:

```bash
git add .
```

To create a commit:

```bash
git commit -m "Update medical chatbot"
```

To push changes:

```bash
git push
```

## Limitations

* The chatbot depends on the information available in the provided medical documents.
* Answers may be incomplete if relevant information is not available in the knowledge base.
* The quality of the response depends on document retrieval and the language model.
* The current system uses a local FAISS database.
* Internet access is required for communication with the Groq API.
* The chatbot should not be treated as a replacement for a qualified medical professional.

## Medical Safety

This project is intended for educational and informational purposes.

The chatbot should not be used as a substitute for professional medical diagnosis, treatment, or emergency medical care.

Users should consult a qualified healthcare professional for personal medical concerns, diagnosis, medication decisions, or treatment recommendations.

## Future Improvements

Possible future improvements include:

* Improved medical document preprocessing
* More medical documents and knowledge sources
* Better retrieval techniques
* Hybrid search
* Reranking retrieved documents
* Conversation-aware retrieval
* Medical source citations
* Improved UI/UX
* User authentication
* Conversation storage
* Voice input and output
* Multilingual support
* Deployment to a cloud platform
* Medical safety and emergency-response handling
* Evaluation using a dedicated medical QA dataset

## Author

**Jalan Bamjan**

GitHub:

```text
https://github.com/jalan538
```

## License

This project is developed for educational and academic purposes.
