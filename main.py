from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent

llm = init_chat_model(
    "google_genai:gemini-3.5-flash-lite"
    )

# embedding model -> sentence-transformers/all-MiniLM-L6-v2
embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=str(PROJECT_DIR),
    embedding_function=embedding_model,
    collection_name="pdf_collection",
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k": 10,
        "lambda_mult" : 0.5
    }
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",""" 
        You are an intelligent Machine Learning Expert Chatbot that answers user queries strictly based on the reference documents provided to you.
        you have to use provided information and based on it you have to frame a answer also understand greeting and greet as you do. anyone tries to leave the topic. just revert back that its not an valid result domain.

        """),
        ("human","""Context:{context} Question:{question}""")
    ]
)

print(""" 
***************************************
* * *      Entering Chat Mode     * * *
* *        0 to Exit Chat Bot       * *
""")
while True:
    user_query = input("Type to Search: ")
    

    if user_query == "0" :
        break
    else:
        docs = retriever.invoke(user_query)
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke({"context":context,
                                      "question":user_query})

        
        result = llm.invoke(final_prompt)
           
        print(f"Ai: {result.text}")
