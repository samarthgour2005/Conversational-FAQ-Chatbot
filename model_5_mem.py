from dotenv import load_dotenv 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.retrievers import MultiQueryRetriever
from qdrant_client import QdrantClient, models
from langchain_qdrant import Qdrant
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser  
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
import cohere
from langchain.memory import ConversationBufferMemory,ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever  
from langchain.retrievers.document_compressors import CohereRerank
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

load_dotenv()
co=cohere.Client()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",google_api_key="AIzaSyAXCOANBR3cvPSi5HYIGWlV4liO_a-pml0")
model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    top_k=5,
    top_p=0.6,
    temperature=0.5,            
)

def get_vectorstore():
    client=QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    vectorstore=Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLECTION_NAME"),
        embeddings=embeddings
    )
    return vectorstore

vectorstore=get_vectorstore()

sources = [
    "0.0_chat_bot_task/Getting_Started_with_Encrypta.pdf",
    "0.0_chat_bot_task\Getting_Started_with_Password_Manager.pdf"
]
# 3. MMR retriever
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2}, 
    lambda_mult=0.3,
    score_threshold=0.3,
)


robust_retriever = MultiQueryRetriever.from_llm(
    retriever=mmr_retriever,
    llm=model,
)

re_ranker = CohereRerank(client=co, model="rerank-v3.5", top_n=5)

rerank_retriever = ContextualCompressionRetriever(
    base_retriever=robust_retriever,
    base_compressor=re_ranker)

def joint_context(inputs):
    contexts = [doc.page_content for doc in inputs]
    combined_context = "\n\n".join(contexts)
    return combined_context



def cohere_rerank(inputs):
    docs = [doc.page_content for doc in inputs]
    results = co.rerank(model="rerank-v3.5", query=query, documents=docs, top_n=len(docs))
    reranked_docs = [inputs[result.index] for result in results.results]
    return reranked_docs




def answer_template(language="english"):
    template = f"""You are a concise FAQ chatbot.
            if user greets you, greet them back politely.but greet only once in the whole conversation. 
            If the question is related to the given context, answer briefly and format the answer in short lines (each main idea on a new line).
            If the question is not related to the context, reply only with: "I don't know about it."

            <context>
            {{chat_history}}
            {{context}}
            </context>

            Question: {{question}}
            Language: {language}.
"""
    return template




#memory
def create_mem():
    memory = ConversationSummaryBufferMemory(
            max_token_limit=2000,
            llm=model,
            return_messages=True,
            memory_key='chat_history',
            output_key="answer",
            input_key="question"
        )
    return memory


def create_ConversationalRetrievalChai():
    memory=create_mem()
    stand_alone_question_prompt= PromptTemplate(
         template="""Given the following conversation and a follow up question, 
                    rephrase the follow up question to be a standalone question, in its original language.\n\n
                    Chat History:\n{chat_history}\n
                    Follow Up Input: {question}\n
                    Standalone question:""",
    input_variables=['chat_history', 'question']
    )

    ans_prompt=ChatPromptTemplate.from_template(answer_template())
    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=stand_alone_question_prompt,
        combine_docs_chain_kwargs={'prompt': ans_prompt},
        condense_question_llm=model,
        memory=memory,
        retriever = rerank_retriever,
        llm=model,
        chain_type= 'stuff', # options- 'stuff', 'map_reduce', 'refine'
        verbose= False,
        return_source_documents=False    
    )

    return chain,memory

chain,memory=create_ConversationalRetrievalChai()

# final_chain=RunnableSequence(chain,StrOutputParser())

while True:
    # Get user input
    query = input("🧑 You: ")
    
    if query.lower() in ['q', 'exit']:
        print("🤖 Bot: Bye! Have a great day! 👋")
        break

    print("🤖 Bot: ", end="", flush=True)

    result=chain.invoke({"question":query})
    print(result['answer'], end="")
    print("\n")  # move to next line after full answer
