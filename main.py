from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma                
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = init_chat_model(
    "google_genai:gemini-3.5-flash-lite"
    )

# embedding model -> sentence-transformers/all-MiniLM-L6-v2
embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chromaDb",
    embedding_function=embedding_model
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

Instructions
Answer questions using only the information available in the provided reference documents.
Do not use external sources, pretrained knowledge, assumptions, or unsupported information.
Ensure every answer is accurate, relevant, and grounded in the retrieved context.
If the answer is not available in the provided references, respond exactly with: "I am not able to answer this at the moment."
Never hallucinate, fabricate information, or make unsupported assumptions.
Use conversation history to understand follow-up questions while keeping all factual answers document-based.
Keep responses clear, concise, and easy to understand.
Primary Objective
Provide reliable, accurate, and context-aware answers related to Machine Learning, strictly following the provided reference documents.
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
