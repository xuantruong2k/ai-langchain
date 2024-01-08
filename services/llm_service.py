from typing import List
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
    )
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.docstore.document import Document
from backports import configparser
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, utility

from messsages.inbound_http_request import ChatRequest
import json

class InternalLlmService:
    _langchain_llm: Ollama
    _embeddings: OllamaEmbeddings

    def __init__(self):
        parser = configparser.ConfigParser()
        parser.read("config.conf")
        prompt = ChatPromptTemplate(
            message = [
                SystemMessagePromptTemplate.from_template(
                    """You are a helpful assistant use {docs} in my system data. Find company information from id"""
                ),
                HumanMessagePromptTemplate.from_template("{question}?")
            ]
        )
        llm = Ollama(
            model=parser["model"]["name"],
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler])
        )
        self._langchain_llm = LLMChain(llm=llm, prompt=prompt)
        self._embeddings = OllamaEmbeddings(model=parser["model"]["name"])

    async def lc_chat(self, request: ChatRequest):
        question_vector: List[float] = await self._embeddings.aembed_query(request.prompt)
        
        # semantic search from vector database
        connections.connect(host="127.0.0.1", port=19530)
        db.using_database("invoice")
        collection = Collection("company")
        collection.load()
        semanticResults = collection.search(
            data=[question_vector],
            anns_field="vector",
            param={},
            limit=5,
            output_fields=["company_id", "company_name", "company_address"],
            consistency_level="Strong"
        )
        collection.release()

        # convert result to json
        docs = []
        for result in semanticResults[0]:
            entity = result.entity
            object_content = {
                "id": entity.fields["company_id"],
                "company_name": entity.fields["company_name"],
                "company_address": entity.fields["company_address"],
            }
            doc = Document(page_content=json.dumps(object_content), metadata={"source":"mongo"})
            docs.append(doc)

        # call to llm
        result = self._langchain_llm(
            {"question": request.promt, "docs": docs}
        )

        # response result
        return {
            "answer": result["text"],
            "question_vector": question_vector
        }
            



        
        
