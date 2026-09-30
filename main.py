import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


load_dotenv(find_dotenv())


DB_FAISS_PATH = "vectorestore/db_faiss"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    return prompt


def main():
    st.title("Medical Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Ask your medical question...")

    if prompt:
        st.chat_message("user").markdown(prompt)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

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

        try:
            vectorstore = get_vectorstore()

            if vectorstore is None:
                st.error("Failed to load the vector store.")
                return

            groq_api_key = os.environ.get("GROQ_API_KEY")

            if not groq_api_key:
                st.error("GROQ_API_KEY is not found. Please check your .env file.")
                return

            llm = ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0.0,
                max_tokens=512,
                groq_api_key=groq_api_key
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
                }
            )

            response = qa_chain.invoke(
                {
                    "query": prompt
                }
            )

            result = response["result"]
            source_documents = response["source_documents"]

            st.chat_message("assistant").markdown(result)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result
                }
            )

            with st.expander("Source Documents"):
                for i, doc in enumerate(source_documents, start=1):
                    st.write(f"Source {i}")
                    st.write(doc.metadata)

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()