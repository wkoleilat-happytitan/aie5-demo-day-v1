import os
from openai import AsyncOpenAI
import chainlit as cl
from dotenv import load_dotenv
from open_deep_research_py import builder, Configuration
import asyncio
import uuid

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_template = """You are a helpful AI assistant. Provide direct answers to user questions without mentioning anything about message formats or code objects."""

@cl.on_chat_start
async def start():
    # Generate a unique session ID
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    await cl.Message(content="Welcome! What topic would you like me to research and generate a report about?").send()

@cl.on_message
async def main(message: str):
    # Initialize progress message
    progress_msg = await cl.Message(content="Starting research process...").send()

    # Create configuration
    config = {
        "configurable": {
            "thread_id": cl.user_session.get("thread_id"),
            "planner_provider": "anthropic",
            "writer_provider": "anthropic",
            "search_api": "tavily"
        }
    }

    try:
        # Run the graph
        graph = builder.compile()
        stream = graph.astream(
            {"topic": message},
            config,
            stream_mode="updates"
        )
        
        try:
            async for chunk in stream:
                if isinstance(chunk, dict):
                    if 'final_report' in chunk:
                        await cl.Message(content=chunk['final_report']).send()
                        break
                    elif '__interrupt__' in chunk:
                        continue
                    else:
                        # Create new message instead of updating
                        await cl.Message(f"Research in progress: {str(chunk)}").send()
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            await stream.aclose()
            raise
        finally:
            # Ensure the stream is properly closed
            await stream.aclose()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()