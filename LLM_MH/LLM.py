import random
import tsplib95
import time
import csv
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage

from langchain.chains import LLMChain
from langchain_community.callbacks.manager import get_openai_callback
from src.config import FLAGS
from datetime import datetime

def llm_process_ec(llm, tokens, secs, fitness, current_population, pragmas_possible_value, result_number):
    system_message = SystemMessage(content=FLAGS.content)
    input_message = HumanMessagePromptTemplate.from_template('''
                current_population: {current_population}
                fitness: {fitness}
                pragmas_possible_value: {pragmas_possible_value}
                result_number: {result_number}
                '''
                                             )
    if llm == "deepseek-r1" and "deepseek-v3":
        template = ChatPromptTemplate.from_messages(
            [{"role":"system", "content":system_message},
            {"role":"user", "content": input_message}]
        )
    else:
        template = ChatPromptTemplate.from_messages(
            [system_message, input_message]
        )

    with get_openai_callback() as cb:
        chain = LLMChain(llm=llm, prompt=template)
        tokens += cb.total_tokens
        if tokens >= 90000:
            time.sleep(62 - secs)
        res = chain.run(fitness=fitness, current_population=current_population, pragmas_possible_value=pragmas_possible_value, result_number=result_number)
    return res, tokens

def llm_process_aco(llm, tokens, secs, fitness, current_population, pragmas_possible_value, result_number, pheromone_matrix):
    system_message = SystemMessage(content=FLAGS.content1)
    input_message = HumanMessagePromptTemplate.from_template('''
                current_population: {current_population}
                fitness: {fitness}
                pragmas_possible_value: {pragmas_possible_value}
                result_number: {result_number}
                pheromone_matrix: {pheromone_matrix}
                '''
                                             )
    if llm == "deepseek-r1" and "deepseek-v3":
        template = ChatPromptTemplate.from_messages(
            [{"role":"system", "content":system_message},
            {"role":"user", "content": input_message}]
        )
    else:
        template = ChatPromptTemplate.from_messages(
            [system_message, input_message]
        )

    with get_openai_callback() as cb:
        chain = LLMChain(llm=llm, prompt=template)
        tokens += cb.total_tokens
        if tokens >= 90000:
            time.sleep(62 - secs)
        res = chain.run(fitness=fitness, current_population=current_population, pragmas_possible_value=pragmas_possible_value, result_number=result_number, pheromone_matrix=pheromone_matrix)
    return res, tokens
