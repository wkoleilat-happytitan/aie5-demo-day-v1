import os
from openai import AsyncOpenAI
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_template = """You are a helpful AI assistant. Provide direct answers to user questions without mentioning anything about message formats or code objects."""

@cl.on_chat_start
async def start():
    # Initialize the message history when chat starts
    cl.user_session.set("messages", [])
    await cl.Message(content="Hello! I'm here to help. What would you like to know?").send()

@cl.on_message
async def main(message: str):
    # Get message history
    messages = cl.user_session.get("messages", [])
    
    # Extract message content
    user_message = message.content if hasattr(message, 'content') else str(message)
    
    # Add user message to history
    messages.append({"role": "user", "content": user_message})
    
    # Prepare messages for API call
    api_messages = [
        {"role": "system", "content": system_template},
        *messages
    ]
    
    # Get response from OpenAI
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=api_messages,
        temperature=0.7,
        stream=True
    )

    # Stream the response
    msg = cl.Message(content="")
    response_content = ""
    
    async for chunk in response:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            response_content += token
            await msg.stream_token(token)
    
    await msg.send()
    
    # Add assistant's response to history
    messages.append({"role": "assistant", "content": response_content})
    cl.user_session.set("messages", messages)