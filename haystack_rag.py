from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
import PyPDF2
import warnings
from haystack.dataclasses import Document
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
import streamlit as st

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()  


def document_store(document):
    document_store = InMemoryDocumentStore()
    splitter = DocumentSplitter(split_by="word", split_length=200)       #splits documents into chunks
    embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small") #creates vector embeddings
    writer = DocumentWriter(document_store=document_store)  #writes embeddings into vector data store

    #creating Pipeline
    indexing_pipeline = Pipeline()

    ######adding components to the pipeline
    #indexing_pipeline.add_component("converter", converter)
    indexing_pipeline.add_component("splitter", splitter)
    indexing_pipeline.add_component("embedder", embedder)
    indexing_pipeline.add_component("writer", writer)

    ######connecting componets
    #indexing_pipeline.connect("converter", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    #indexing_pipeline.run({"converter": {"sources": [r'C:\Users\Ivana\Desktop\CODEProjects\LLM.HRResumeAnalyser\CV_IvaMatic_with recommendations .pdf']}})
    indexing_pipeline.run({"documents": document})
    ##if converter used in pipeline
    ##indexing_pipeline.run({"converter": {"sources":document}})
    print("Number of documents:",document_store.count_documents()) #in prompt we need to iterate over the documents that retriever will select!
    ##checking the document store at index 1
    #print(document_store.filter_documents()[1].content)
    return document_store

def rag_pipeline(document,query):
    query_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
    retriever = InMemoryEmbeddingRetriever(document_store=document_store(document))
    prompt_builder = PromptBuilder(template=prompt)
    generator = OpenAIGenerator(model="gpt-4o")

    ##creating a pipeline
    rag = Pipeline()
    #####adding components to the pipeline
    rag.add_component("query_embedder", query_embedder)
    rag.add_component("retriever", retriever)
    rag.add_component("prompt", prompt_builder)
    rag.add_component("generator", generator)

    #####connecting components in the pipeline
    rag.connect("query_embedder.embedding", "retriever.query_embedding")
    rag.connect("retriever.documents", "prompt.documents")
    rag.connect("prompt", "generator")


    result = rag.run(
        {
            "query_embedder": {"text": query},
            "retriever": {"top_k": 4},
            "prompt": {"query": query},
        }
    )

    print(result)
    return result

## Retrieval Augmented Generation with Prompt
prompt = """
    Your task is to help the recruiter to get to know the job candidate better. 
    You will be provided with Context which is a candidates Resume. 
    Answer the recruiter questions based on the provided Context. 
    Don't put yourself in front, answer based on the Context.
    When enough informations provided give longer and informative answers. Write it in a structured form that is easier to read. 
    If there are not information in the Context, recommend to contact the job candidate through his/hers email.

    Context:
    {% for doc in documents %}
        {{ doc.content }} 

    {% endfor %}

    Question: {{ query }} 

    Answer: 
    """

# if submitted:
#     result = rag_pipeline()
#     st.info(result['generator']['replies'][0])