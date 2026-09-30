import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv, find_dotenv


# Load environment variables
load_dotenv(find_dotenv())


# Step 1: Setup Groq LLM

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def load_llm():
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.5,
        max_tokens=512,
        api_key=GROQ_API_KEY
    )

    return llm


# Step 2: Create custom prompt

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.

If you don't know the answer, just say that you don't know.
Do not make up an answer.
Only use information from the provided context.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk please.
"""


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    return prompt


# Step 3: Load FAISS database

DB_FAISS_PATH = "vectorestore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# Step 4: Create Retrieval QA chain

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(
        search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    }
)


# Step 5: Ask user for question

user_query = input("Write Query Here: ")

response = qa_chain.invoke({
    "query": user_query
})


print("\nRESULT:")
print(response["result"])


print("\nSOURCE DOCUMENTS:")

for doc in response["source_documents"]:
    print(doc.metadata)