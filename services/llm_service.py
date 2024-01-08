from typing import List
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
    )
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from backports import configparser

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

    async def lc_chat(self, request: ChatRequest):
