from modules.environment.environment_utilities import (
    load_environment_variables,
    verify_environment_variables,
)
from modules.neo4j.credentials import neo4j_credentials
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_memory(session_id):
    return memory

# Main program
try:
    # Load environment variables using the utility
    env_vars = load_environment_variables()
    

    # Verify the environment variables
    if not verify_environment_variables(env_vars):
        raise ValueError("Some environment variables are missing!")

    chat_llm = OpenAI(
        openai_api_key=env_vars["OPEN_AI_SECRET_KEY"],
        model="gpt-3.5-turbo-instruct",
        temperature=0)
  
    instructions = SystemMessage(content="""
                                        You are a surfer dude, having a conversation about the surf conditions on the beach.
                                        Respond using surfer slang.
                                        """)

    question = HumanMessage(content="What is the weather like?")

    # response = chat_llm.invoke([
    #                 instructions,
    #                 question
    #             ])

    # print(response.content)

    # Ara usant el prompt i chains
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            {instructions},
        ),
        (
            "system", 
            "{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human", 
            "{question}"
        ),
    ])

    memory = ChatMessageHistory()

    chat_chain = prompt | chat_llm | StrOutputParser()
    
    chat_with_message_history = RunnableWithMessageHistory(
        chat_chain,
        get_memory,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # afegim context de l'estat de les platges
    current_weather = """
        {
            "surf": [
                {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
                {"beach": "Polzeath", "conditions": "Flat and calm"},
                {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
            ]
        }"""

    # response = chat_chain.invoke({"question": "What is the weather like?"})
    # Ara afegim el context per tenir growing data
    response = chat_chain.invoke(
        {
            "context": current_weather,
            "question": "What is the weather like?"
        })
    print(response)

    # Ara incloem memoritzar l'anterior context
    response = chat_with_message_history.invoke(
    {
        "context": current_weather,
        "question": "Hi, I am at Watergate Bay. What is the surf like?"
    },

    config={"configurable": {"session_id": "none"}})
    print(response)

    response = chat_with_message_history.invoke(
        {
            "context": current_weather,
            "question": "Where I am?"
        },
        config={"configurable": {"session_id": "none"}}
    )
    print(response)

except Exception as e:
    print(f"An unexpected error occurred: {e}")