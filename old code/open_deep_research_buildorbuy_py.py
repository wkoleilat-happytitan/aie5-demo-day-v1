# Standard library imports
import os
import uuid
import asyncio
import requests
from enum import Enum
from dataclasses import dataclass, fields
from typing import Optional, Dict, Any, Literal, TypeVar, Annotated, List, TypedDict
from dotenv import load_dotenv 

# Remove IPython import and add fallback display function
def display_markdown(text: str, title: str = ""):
    """Display markdown text, falling back to plain text if IPython is not available"""
    try:
        from IPython.display import Markdown, display
        display(Markdown(f"# {title}\n\n{text}"))
    except ImportError:
        print(f"\n{title}")
        print("=" * 80)
        print(text)

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langgraph.graph import StateGraph, END, START
from typing import TypeVar, Annotated, List

# LangGraph imports
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send

# Websearch imports
from tavily import TavilyClient, AsyncTavilyClient
from exa_py import Exa

# Local imports
from capability_extractor import extract_capability, select_file, create_retriever_from_file

# New import
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langsmith import traceable

# Define state types here instead of importing State
StateType = TypeVar("StateType")

load_dotenv()


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")


from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
import operator

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the build or buy analysis covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )   

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")
    
    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": False
    }

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )
    
    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": False
    }

class Feedback(BaseModel):
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail').",
        pattern="^(pass|fail)$"  # Add pattern constraint
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
        default_factory=list,
        min_items=0,  # Add validation
        max_items=5   # Add validation
    )
    
    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": False,
        "json_schema_extra": {
            "examples": [{
                "grade": "pass",
                "follow_up_queries": []
            }]
        },
        "strict": True  # Make the model strict
    }

    @classmethod
    def pass_feedback(cls) -> "Feedback":
        """Create a passing feedback with no follow-up queries"""
        return cls(grade="pass", follow_up_queries=[])

    @classmethod
    def fail_feedback(cls, queries: List[SearchQuery]) -> "Feedback":
        """Create a failing feedback with follow-up queries"""
        return cls(grade="fail", follow_up_queries=queries)

    def __str__(self) -> str:
        """String representation for logging"""
        return f"Feedback(grade={self.grade}, queries={len(self.follow_up_queries)})"

class ReportStateInput(TypedDict):
    capability: str # Report topic

class ReportStateOutput(TypedDict):
    final_report: str # Final report

class ReportState(TypedDict):
    capability: str # Report topic    
    feedback_on_report_plan: str # Feedback on the report plan
    sections: list[Section] # List of report sections 
    completed_sections: Annotated[list, operator.add] # Send() API key
    report_sections_from_research: str # String of any completed sections from research to write final sections
    final_report: str # Final report

class SectionState(TypedDict):
    capability: str # Report topic
    section: Section # Report section  
    search_iterations: int # Number of search iterations done
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    report_sections_from_research: str # String of any completed sections from research to write final sections
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API


tavily_client = TavilyClient()
tavily_async_client = AsyncTavilyClient()


def get_config_value(value):
    """
    Helper function to handle both string and enum cases of configuration values
    """
    return value if isinstance(value, str) else value.value


# Helper function to get search parameters based on the search API and config
def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters the search_api_config dictionary to include only parameters accepted by the specified search API.

    Args:
        search_api (str): The search API identifier (e.g., "exa", "tavily").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    # Define accepted parameters for each search API
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": [],  # Tavily currently accepts no additional parameters
        "perplexity": [],  # Perplexity accepts no additional parameters
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}


def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=False):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source.

    Args:
        search_responses: List of search response dicts, each containing:
            - query: str
            - results: List of dicts with fields:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
        include_raw_content: bool

    Returns:
        str: Formatted string with deduplicated sources
    """
     # Collect all results
    sources_list = []
    for response in search_response:
        sources_list.extend(response['results'])

    # Deduplicate by URL
    unique_sources = {source['url']: source for source in sources_list}

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()


def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str


@traceable
async def tavily_search_async(search_queries):
    """
    Performs concurrent web searches using the Tavily API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process

    Returns:
            List[dict]: List of search responses from Tavily API, one per query. Each response has format:
                {
                    'query': str, # The original search query
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': list,
                    'results': [                     # List of search results
                        {
                            'title': str,            # Title of the webpage
                            'url': str,              # URL of the result
                            'content': str,          # Summary/snippet of content
                            'score': float,          # Relevance score
                            'raw_content': str|None  # Full page content if available
                        },
                        ...
                    ]
                }
    """

    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=5,
                    include_raw_content=True,
                    topic="general"
                )
            )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    return search_docs


@traceable
def perplexity_search(search_queries):
    """Search the web using the Perplexity API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process

    Returns:
        List[dict]: List of search responses from Perplexity API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }

    search_docs = []
    for query in search_queries:

        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "Search the web and provide factual information with sources."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        }

        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse the response
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", ["https://perplexity.ai"])

        # Create results list for this query
        results = []

        # First citation gets the full content
        results.append({
            "title": f"Perplexity Search, Source 1",
            "url": citations[0],
            "content": content,
            "raw_content": content,
            "score": 1.0  # Adding score to match Tavily format
        })

        # Add additional citations without duplicating content
        for i, citation in enumerate(citations[1:], start=2):
            results.append({
                "title": f"Perplexity Search, Source {i}",
                "url": citation,
                "content": "See primary source for full content",
                "raw_content": None,
                "score": 0.5  # Lower score for secondary sources
            })

        # Format response to match Tavily structure
        search_docs.append({
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": [],
            "results": results
        })

    return search_docs


@traceable
async def exa_search(search_queries, max_characters: Optional[int] = None, num_results=5, 
                     include_domains: Optional[List[str]] = None, 
                     exclude_domains: Optional[List[str]] = None,
                     subpages: Optional[int] = None):
    """Search the web using the Exa API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        max_characters (int, optional): Maximum number of characters to retrieve for each result's raw content.
                                       If None, the text parameter will be set to True instead of an object.
        num_results (int): Number of search results per query. Defaults to 5.
        include_domains (List[str], optional): List of domains to include in search results. 
            When specified, only results from these domains will be returned.
        exclude_domains (List[str], optional): List of domains to exclude from search results.
            Cannot be used together with include_domains.
        subpages (int, optional): Number of subpages to retrieve per result. If None, subpages are not retrieved.

    Returns:
        List[dict]: List of search responses from Exa API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """
    # Check that include_domains and exclude_domains are not both specified
    if include_domains and exclude_domains:
        raise ValueError("Cannot specify both include_domains and exclude_domains")

    # Initialize Exa client (API key should be configured in your .env file)
    exa = Exa(api_key = f"{os.getenv('EXA_API_KEY')}")

    # Define the function to process a single query
    async def process_query(query):
        # Use run_in_executor to make the synchronous exa call in a non-blocking way
        loop = asyncio.get_event_loop()

        # Define the function for the executor with all parameters
        def exa_search_fn():
            # Build parameters dictionary
            kwargs = {
                # Set text to True if max_characters is None, otherwise use an object with max_characters
                "text": True if max_characters is None else {"max_characters": max_characters},
                "summary": True,  # This is an amazing feature by EXA. It provides an AI generated summary of the content based on the query
                "num_results": num_results
            }

            # Add optional parameters only if they are provided
            if subpages is not None:
                kwargs["subpages"] = subpages

            if include_domains:
                kwargs["include_domains"] = include_domains
            elif exclude_domains:
                kwargs["exclude_domains"] = exclude_domains

            return exa.search_and_contents(query, **kwargs)

        response = await loop.run_in_executor(None, exa_search_fn)

        # Format the response to match the expected output structure
        formatted_results = []
        seen_urls = set()  # Track URLs to avoid duplicates

        # Helper function to safely get value regardless of if item is dict or object
        def get_value(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            else:
                return getattr(item, key, default) if hasattr(item, key) else default

        # Access the results from the SearchResponse object
        results_list = get_value(response, 'results', [])

        # First process all main results
        for result in results_list:
            # Get the score with a default of 0.0 if it's None or not present
            score = get_value(result, 'score', 0.0)

            # Combine summary and text for content if both are available
            text_content = get_value(result, 'text', '')
            summary_content = get_value(result, 'summary', '')

            content = text_content
            if summary_content:
                if content:
                    content = f"{summary_content}\n\n{content}"
                else:
                    content = summary_content

            title = get_value(result, 'title', '')
            url = get_value(result, 'url', '')

            # Skip if we've seen this URL before (removes duplicate entries)
            if url in seen_urls:
                continue

            seen_urls.add(url)

            # Main result entry
            result_entry = {
                "title": title,
                "url": url,
                "content": content,
                "score": score,
                "raw_content": text_content
            }

            # Add the main result to the formatted results
            formatted_results.append(result_entry)

        # Now process subpages only if the subpages parameter was provided
        if subpages is not None:
            for result in results_list:
                subpages_list = get_value(result, 'subpages', [])
                for subpage in subpages_list:
                    # Get subpage score
                    subpage_score = get_value(subpage, 'score', 0.0)

                    # Combine summary and text for subpage content
                    subpage_text = get_value(subpage, 'text', '')
                    subpage_summary = get_value(subpage, 'summary', '')

                    subpage_content = subpage_text
                    if subpage_summary:
                        if subpage_content:
                            subpage_content = f"{subpage_summary}\n\n{subpage_content}"
                        else:
                            subpage_content = subpage_summary

                    subpage_url = get_value(subpage, 'url', '')

                    # Skip if we've seen this URL before
                    if subpage_url in seen_urls:
                        continue

                    seen_urls.add(subpage_url)

                    formatted_results.append({
                        "title": get_value(subpage, 'title', ''),
                        "url": subpage_url,
                        "content": subpage_content,
                        "score": subpage_score,
                        "raw_content": subpage_text
                    })

        # Collect images if available (only from main results to avoid duplication)
        images = []
        for result in results_list:
            image = get_value(result, 'image')
            if image and image not in images:  # Avoid duplicate images
                images.append(image)

        return {
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": images,
            "results": formatted_results
        }

    # Process all queries sequentially with delay to respect rate limit
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (0.25s = 4 requests per second, well within the 5/s limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(0.25)

            result = await process_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing query '{query}': {str(e)}")
            # Add a placeholder result for failed queries to maintain index alignment
            search_docs.append({
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e)
            })

            # Add additional delay if we hit a rate limit error
            if "429" in str(e):
                print("Rate limit exceeded. Adding additional delay...")
                await asyncio.sleep(1.0)  # Add a longer delay if we hit a rate limit

    return search_docs


@traceable
async def arxiv_search_async(search_queries, load_max_docs=5, get_full_documents=True, load_all_available_meta=True):
    """
    Performs concurrent searches on arXiv using the ArxivRetriever.

    Args:
        search_queries (List[str]): List of search queries or article IDs
        load_max_docs (int, optional): Maximum number of documents to return per query. Default is 5.
        get_full_documents (bool, optional): Whether to fetch full text of documents. Default is True.
        load_all_available_meta (bool, optional): Whether to load all available metadata. Default is True.

    Returns:
        List[dict]: List of search responses from arXiv, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL (Entry ID) of the paper
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str|None  # Full paper content if available
                    },
                    ...
                ]
            }
    """

    async def process_single_query(query):
        try:
            # Create retriever for each query
            retriever = ArxivRetriever(
                load_max_docs=load_max_docs,
                get_full_documents=get_full_documents,
                load_all_available_meta=load_all_available_meta
            )

            # Run the synchronous retriever in a thread pool
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: retriever.invoke(query))

            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0

            for i, doc in enumerate(docs):
                # Extract metadata
                metadata = doc.metadata

                # Use entry_id as the URL (this is the actual arxiv link)
                url = metadata.get('entry_id', '')

                # Format content with all useful metadata
                content_parts = []

                # Primary information
                if 'Summary' in metadata:
                    content_parts.append(f"Summary: {metadata['Summary']}")

                if 'Authors' in metadata:
                    content_parts.append(f"Authors: {metadata['Authors']}")

                # Add publication information
                published = metadata.get('Published')
                published_str = published.isoformat() if hasattr(published, 'isoformat') else str(published) if published else ''
                if published_str:
                    content_parts.append(f"Published: {published_str}")

                # Add additional metadata if available
                if 'primary_category' in metadata:
                    content_parts.append(f"Primary Category: {metadata['primary_category']}")

                if 'categories' in metadata and metadata['categories']:
                    content_parts.append(f"Categories: {', '.join(metadata['categories'])}")

                if 'comment' in metadata and metadata['comment']:
                    content_parts.append(f"Comment: {metadata['comment']}")

                if 'journal_ref' in metadata and metadata['journal_ref']:
                    content_parts.append(f"Journal Reference: {metadata['journal_ref']}")

                if 'doi' in metadata and metadata['doi']:
                    content_parts.append(f"DOI: {metadata['doi']}")

                # Get PDF link if available in the links
                pdf_link = ""
                if 'links' in metadata and metadata['links']:
                    for link in metadata['links']:
                        if 'pdf' in link:
                            pdf_link = link
                            content_parts.append(f"PDF: {pdf_link}")
                            break

                # Join all content parts with newlines 
                content = "\n".join(content_parts)

                result = {
                    'title': metadata.get('Title', ''),
                    'url': url,  # Using entry_id as the URL
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.page_content if get_full_documents else None
                }
                results.append(result)

            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing arXiv query '{query}': {str(e)}")
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }

    # Process queries sequentially with delay to respect arXiv rate limit (1 request per 3 seconds)
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (3 seconds per ArXiv's rate limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(3.0)

            result = await process_single_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing arXiv query '{query}': {str(e)}")
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })

            # Add additional delay if we hit a rate limit error
            if "429" in str(e) or "Too Many Requests" in str(e):
                print("ArXiv rate limit exceeded. Adding additional delay...")
                await asyncio.sleep(5.0)  # Add a longer delay if we hit a rate limit

    return search_docs


@traceable
async def pubmed_search_async(search_queries, top_k_results=5, email=None, api_key=None, doc_content_chars_max=4000):
    """
    Performs concurrent searches on PubMed using the PubMedAPIWrapper.

    Args:
        search_queries (List[str]): List of search queries
        top_k_results (int, optional): Maximum number of documents to return per query. Default is 5.
        email (str, optional): Email address for PubMed API. Required by NCBI.
        api_key (str, optional): API key for PubMed API for higher rate limits.
        doc_content_chars_max (int, optional): Maximum characters for document content. Default is 4000.

    Returns:
        List[dict]: List of search responses from PubMed, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL to the paper on PubMed
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str       # Full abstract content
                    },
                    ...
                ]
            }
    """

    async def process_single_query(query):
        try:
            # print(f"Processing PubMed query: '{query}'")

            # Create PubMed wrapper for the query
            wrapper = PubMedAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                email=email if email else "your_email@example.com",
                api_key=api_key if api_key else ""
            )

            # Run the synchronous wrapper in a thread pool
            loop = asyncio.get_event_loop()

            # Use wrapper.lazy_load instead of load to get better visibility
            docs = await loop.run_in_executor(None, lambda: list(wrapper.lazy_load(query)))

            print(f"Query '{query}' returned {len(docs)} results")

            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0

            for i, doc in enumerate(docs):
                # Format content with metadata
                content_parts = []

                if doc.get('Published'):
                    content_parts.append(f"Published: {doc['Published']}")

                if doc.get('Copyright Information'):
                    content_parts.append(f"Copyright Information: {doc['Copyright Information']}")

                if doc.get('Summary'):
                    content_parts.append(f"Summary: {doc['Summary']}")

                # Generate PubMed URL from the article UID
                uid = doc.get('uid', '')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""

                # Join all content parts with newlines
                content = "\n".join(content_parts)

                result = {
                    'title': doc.get('Title', ''),
                    'url': url,
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.get('Summary', '')
                }
                results.append(result)

            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # Handle exceptions with more detailed information
            error_msg = f"Error processing PubMed query '{query}': {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())  # Print full traceback for debugging

            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }

    # Process all queries with a reasonable delay between them
    search_docs = []

    # Start with a small delay that increases if we encounter rate limiting
    delay = 1.0  # Start with a more conservative delay

    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests
            if i > 0:  # Don't delay the first request
                # print(f"Waiting {delay} seconds before next query...")
                await asyncio.sleep(delay)

            result = await process_single_query(query)
            search_docs.append(result)

            # If query was successful with results, we can slightly reduce delay (but not below minimum)
            if result.get('results') and len(result['results']) > 0:
                delay = max(0.5, delay * 0.9)  # Don't go below 0.5 seconds

        except Exception as e:
            # Handle exceptions gracefully
            error_msg = f"Error in main loop processing PubMed query '{query}': {str(e)}"
            print(error_msg)

            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })

            # If we hit an exception, increase delay for next query
            delay = min(5.0, delay * 1.5)  # Don't exceed 5 seconds

    return search_docs


def init_chat_model(model: str, model_provider: str, temperature: float = 0) -> ChatOpenAI:
    """Initialize a chat model with the specified configuration."""
    try:
        chat_model = ChatOpenAI(
            model=model,
            temperature=temperature,
            model_kwargs={"response_format": {"type": "text"}}  # Explicitly set response format
        )
        return chat_model
    except Exception as e:
        print(f"Error initializing chat model: {str(e)}")
        print(f"Model: {model}, Provider: {model_provider}, Temperature: {temperature}")
        raise


DEFAULT_REPORT_STRUCTURE = """Use this structure to create an analysis report for build or buy decision of a capability:

1. Introduction (no research needed)
   - Brief overview of the capability

2. Main Body Sections:
   - One section that focused on the buy options for the capability
   - One section that focuss on the build options for the capability
   - One section that compares the buy and build options
   

3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report

Provide a paragraph with no more than 500 words to describe the key take aways on the analysis of the build or buy decision"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"

class PlannerProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"

class WriterProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    report_structure: str = DEFAULT_REPORT_STRUCTURE 
    number_of_queries: int = 2 
    max_search_depth: int = 2 
    planner_provider: PlannerProvider = PlannerProvider.OPENAI
    planner_model: str = "gpt-4o-2024-08-06"  # Changed from Claude to GPT-4
    writer_provider: WriterProvider = WriterProvider.OPENAI
    writer_model: str = "gpt-4o-2024-08-06"
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})


# Prompt to generate search queries to help with planning the report
report_planner_query_writer_instructions="""You are performing an analysis of a build or buy decision report for a capability.

<Capability>
{capability}
</Capability>

<Report organization>
{report_organization}
</Report organization>

<Task>
Your goal is to generate {number_of_queries} web search queries that will help gather information for planning the analysis report of the build or buy decision. 

The queries should:

1. Be related to the Capability
2. Help satisfy the requirements specified in the report organization

Make the queries specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.
</Task>
"""

# Prompt to generate the report plan
report_planner_instructions="""I want a plan for an analysis report that is concise and focused.

<Capability>
The capability that I am considering to build or buy is:
{capability}
</Capability>

<Report organization>
The report should follow this organization: 
{report_organization}
</Report organization>

<Context>
Here is context to use to plan the sections of the report: 
{context}
</Context>

<Task>
Generate a list of sections for the analysis report of the build or buy decision. Your plan should be tight and focused with NO overlapping sections or unnecessary filler. 

For example, a good report structure might look like:
1/ intro
2/ overview section of buy options A
4/ overview section of build options B
4/ comparison between A and B
5/ conclusion

Each section should have the fields:

- Name - Name for this section of the report.
- Description - Brief overview of the build or buy options covered in this section.
- Research - Whether to perform web research for this section of the report.
- Content - The content of the section, which you will leave blank for now.

Integration guidelines:
- Include examples and implementation details within main build or buy options sections, not as separate sections
- Ensure each section has a distinct purpose with no content overlap
- Combine related concepts rather than separating them

Before submitting, review your structure to ensure it has no redundant sections and follows a logical flow.
</Task>

<Feedback>
Here is feedback on the report structure from review (if any):
{feedback}
</Feedback>
"""

# Query writer instructions
query_writer_instructions="""You are a business leader crafting targeted web search queries that will gather comprehensive information for writing a section of an analysis report of a build or buy decision.

<Capability>
{capability}
</Capability>

<Section topic>
{section_topic}
</Section topic>

<Task>
Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information above the section topic. 

The queries should:

1. Be related to the Capability
2. Examine different aspects of the Capability

Make the queries specific enough to find high-quality, relevant sources.
</Task>
"""

# Section writer instructions
section_writer_instructions = """
You are writing a section of a build vs buy analysis report for {capability}.
Section name: {section_name}
Section topic: {section_topic}

Use the following sources to write the section:
{context}

Previous section content (if any):
{section_content}

Write a clear, well-structured section that analyzes the topic based on the provided sources.
Focus on factual information and provide specific details from the sources.
"""

# Instructions for section grading
section_grader_instructions = """
You are evaluating a section of a build vs buy analysis report for {capability}.
Section topic: {section_topic}

Section content:
{section}

Grade this section as 'pass' or 'fail' and provide up to {number_of_follow_up_queries} specific search queries 
for any missing information. If the section is comprehensive, return 'pass' with no follow-up queries.

Evaluate based on:
1. Completeness of analysis
2. Use of specific facts and details
3. Clear comparison of build vs buy options
4. Coverage of key considerations
"""

final_section_writer_instructions = """
You are writing a final section of a build vs buy analysis report for {capability}.
Section name: {section_name}
Section topic: {section_topic}

Use the following completed sections as context:
{context}

Write a clear, well-structured section that builds on the research and analysis from previous sections.
Focus on synthesizing insights and providing clear recommendations.
"""


# Nodes
async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """ Generate the report plan """

    # Inputs
    capability = state["capability"]
    feedback = state.get("feedback_on_report_plan", None)

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Set writer model (model used for query writing and section writing)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, temperature=0) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(capability=capability, report_organization=report_structure, number_of_queries=number_of_queries)

    # Generate queries  
    results = structured_llm.invoke([SystemMessage(content=system_instructions_query),
                                     HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])

    # Web search
    query_list = [query.search_query for query in results.queries]

    # Search the web with parameters
    if search_api == "tavily":
        search_results = await tavily_search_async(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    elif search_api == "perplexity":
        search_results = perplexity_search(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    elif search_api == "exa":
        search_results = await exa_search(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    elif search_api == "arxiv":
        search_results = await arxiv_search_async(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    elif search_api == "pubmed":
        search_results = await pubmed_search_async(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    else:
        raise ValueError(f"Unsupported search API: {search_api}")

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(capability=capability, report_organization=report_structure, context=source_str, feedback=feedback)

    # Set the planner
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    # Report planner instructions
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, plan, research, and content fields."""

    # Run the planner
    if planner_model == "claude-3-7-sonnet-latest-thinking":

        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        planner_llm = init_chat_model(model="claude-3-7-sonnet-latest", 
                                      model_provider=planner_provider, 
                                      max_tokens=20_000, 
                                      thinking={"type": "enabled", "budget_tokens": 16_000})

        # with_structured_output uses forced tool calling, which thinking mode with Claude 3.7 does not support
        # So, we use bind_tools without enforcing tool calling to generate the report sections
        report_sections = planner_llm.bind_tools([Sections]).invoke([SystemMessage(content=system_instructions_sections),
                                                                     HumanMessage(content=planner_message)])
        tool_call = report_sections.tool_calls[0]['args']
        report_sections = Sections.model_validate(tool_call)

    else:

        # With other models, we can use with_structured_output
        planner_llm = init_chat_model(model=planner_model, model_provider=planner_provider)
        structured_llm = planner_llm.with_structured_output(Sections)
        report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections),
                                                 HumanMessage(content=planner_message)])

    # Get sections
    sections = report_sections.sections

    return {"sections": sections}


def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """ Get feedback on the report plan """

    # Get sections
    capability = state["capability"]     # Get system from state
    sections = state['sections']
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    # Get feedback on the report plan from interrupt
    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\n{sections_str}\n\n
                        \nDoes the report plan meet your needs? Pass 'true' to approve the report plan or provide feedback to regenerate the report plan:"""

    feedback = interrupt(interrupt_message)

    # If the user approves the report plan, kick off section writing
    if isinstance(feedback, bool) and feedback is True:
        # Treat this as approve and kick off section writing
        return Command(goto=[
            Send("build_section_with_web_research", {"capability": capability, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research   
        ])

    # If the user provides feedback, regenerate the report plan 
    elif isinstance(feedback, str):
        # Treat this as feedback
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": feedback})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")


def generate_queries(state: SectionState, config: RunnableConfig):
    """ Generate search queries for a report section """

    # Get state 
    capability = state["capability"]
    section = state["section"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, temperature=0) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(capability=capability, 
                                                           section_topic=section.description, 
                                                           number_of_queries=number_of_queries)

    # Generate queries  
    queries = structured_llm.invoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}


async def search_web(state: SectionState, config: RunnableConfig):
    """ Search the web for each query, then return a list of raw sources and a formatted string of sources."""
    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    if search_api == "tavily":
        search_results = await tavily_search_async(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=5000, include_raw_content=True)
    elif search_api == "perplexity":
        search_results = perplexity_search(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=5000, include_raw_content=False)
    elif search_api == "exa":
        search_results = await exa_search(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    elif search_api == "arxiv":
        search_results = await arxiv_search_async(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    elif search_api == "pubmed":
        search_results = await pubmed_search_async(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    else:
        raise ValueError(f"Unsupported search API: {search_api}")

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}


def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """ Write a section of the report """

    # Get state 
    capability = state["capability"]
    section = state["section"]
    source_str = state["source_str"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Format system instructions
    system_instructions = section_writer_instructions.format(capability=capability, 
                                                             section_name=section.name, 
                                                             section_topic=section.description, 
                                                             context=source_str, 
                                                             section_content=section.content)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, temperature=0) 
    section_content = writer_model.invoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])

    # Write content to the section object  
    section.content = section_content.content

    # Grade prompt 
    section_grader_message = """Grade the report and consider follow-up questions for missing information.
                               If the grade is 'pass', return empty strings for all follow-up queries.
                               If the grade is 'fail', provide specific search queries to gather missing information."""

    section_grader_instructions_formatted = section_grader_instructions.format(capability=capability, 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=configurable.number_of_queries)

    # Use planner model for reflection
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    # Update reflection model initialization
    reflection_model = init_chat_model(
        model=planner_model, 
        model_provider=planner_provider,
        temperature=0
    ).with_structured_output(
        Feedback,
        method="function_calling"  # Explicitly set method to function_calling
    )

    feedback = reflection_model.invoke([
        SystemMessage(content=section_grader_instructions_formatted),
        HumanMessage(content=section_grader_message)
    ])

    # If the section is passing or the max search depth is reached, publish the section to completed sections 
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # Publish the section to completed sections 
        return  Command(
        update={"completed_sections": [section]},
        goto=END
    )
    # Update the existing section with new content and update search queries
    else:
        return  Command(
        update={"search_queries": feedback.follow_up_queries, "section": section},
        goto="search_web"
        )


def write_final_sections(state: SectionState, config: RunnableConfig):
    """ Write final sections of the report, which do not require web search and use the completed sections as context """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get state 
    capability = state["capability"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]

    # Format system instructions
    system_instructions = final_section_writer_instructions.format(capability=capability, section_name=section.name, section_topic=section.description, context=completed_report_sections)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, temperature=0) 
    section_content = writer_model.invoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])

    # Write content to section 
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}


def gather_completed_sections(state: ReportState):
    """ Gather completed sections from research and format them as context for writing the final sections """    

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}


def initiate_final_section_writing(state: ReportState):
    """ Write any final sections using the Send API to parallelize the process """    

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {"capability": state["capability"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]


def compile_final_report(state: ReportState):
    """ Compile the final report """    

    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    return {"final_report": all_sections}


section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph -- 

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)


# Create a memory saver for checkpointing
memory = MemorySaver()

# Compile the graph with the checkpointer
graph_with_checkpoint = builder.compile(checkpointer=memory)

thread_id = str(uuid.uuid4())

# Start the graph execution with the topic and display the final report when it appears
async def run_graph_and_show_report(capability: str, auto_approve_plan: bool = True):
    """Run the graph and display the final report when it appears"""
    main_stream = None
    nested_stream = None
    try:
        print(f"Starting graph execution for capability: {capability}")  # Debug log
        
        # Add configuration validation
        config = {
            "configurable": {
                "thread_id": thread_id,
                "planner_provider": "openai",
                "writer_provider": "openai",
                "search_api": "tavily",
                "planner_model": "gpt-4o-2024-08-06",  # Updated model name
                "writer_model": "gpt-4o-2024-08-06",   # Updated model name
                "method": "function_calling",
                "use_anthropic": False,
                "model_provider": "openai",
                "research_model": "gpt-4o-2024-08-06",  # Updated model name
                "disable_anthropic": True,
                "number_of_queries": 2,
                "max_search_depth": 2,
                "report_structure": DEFAULT_REPORT_STRUCTURE
            }
        }
        
        print("Using configuration:", config)  # Debug log
        
        try:
            main_stream = graph_with_checkpoint.astream(
                {"capability": capability}, 
                config,
                stream_mode="updates"
            )
        except Exception as e:
            print(f"Error creating main stream: {str(e)}")
            raise
            
        final_report = None
        async for chunk in main_stream:
            print(f"Processing chunk: {chunk}")  # More detailed logging
            
            if isinstance(chunk, dict):
                if 'final_report' in chunk:
                    final_report = chunk
                    break
                elif 'compile_final_report' in chunk and 'final_report' in chunk['compile_final_report']:
                    final_report = {'final_report': chunk['compile_final_report']['final_report']}
                    break
                elif '__interrupt__' in chunk and auto_approve_plan:
                    try:
                        # Store the nested generator
                        nested_stream = graph_with_checkpoint.astream(
                            Command(resume=True),
                            {"configurable": {"thread_id": thread_id}},
                            stream_mode="updates"
                        )
                        
                        async for response in nested_stream:
                            print(f"Received response: {response}")  # Debug logging
                            
                            if isinstance(response, dict):
                                if 'final_report' in response:
                                    final_report = response
                                    break
                                elif 'compile_final_report' in response and 'final_report' in response['compile_final_report']:
                                    final_report = {'final_report': response['compile_final_report']['final_report']}
                                    break
                    except Exception as e:
                        print(f"Error in nested stream: {str(e)}")
                        raise
                    finally:
                        # Close the nested stream if it exists
                        if nested_stream:
                            await nested_stream.aclose()
                            
        if final_report:
            display_markdown(final_report, "DeepSeek-R1 Report")
            return final_report
        raise Exception("No final report was generated")
                            
    except Exception as e:
        print(f"Error in main stream: {str(e)}")
        raise
    finally:
        # Close the main stream if it exists
        if main_stream:
            await main_stream.aclose()


async def approve_plan():
    """Approve the plan and continue execution with rate limit handling"""
    base_delay = 10 # Start with 1 second delay
    max_delay = 32  # Maximum delay of 32 seconds
    max_retries = 5  # Maximum number of retries
    current_retry = 0

    while current_retry < max_retries:
        try:
            async for chunk in graph_with_checkpoint.astream(
                Command(resume=True), 
                {"configurable": {"thread_id": thread_id}},
                stream_mode="updates"
            ):
                print(chunk)
                print("\n")

                if isinstance(chunk, dict) and 'compile_final_report' in chunk:
                    if 'final_report' in chunk['compile_final_report']:
                        print("🎉 Final report generated! 🎉")
                        final_report = chunk['compile_final_report']['final_report']
                        display_markdown(final_report, "DeepSeek-R1 Report")
                        return
            return  # Success - exit the retry loop

        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                if current_retry < max_retries - 1:
                    delay = min(max_delay, base_delay * (2 ** current_retry))  # Exponential backoff
                    print(f"Rate limit hit. Waiting {delay} seconds before retry {current_retry + 1}/{max_retries}...")
                    await asyncio.sleep(delay)
                    current_retry += 1
                    continue
            # If it's not a rate limit error or we're out of retries, raise the exception
            raise e 


async def provide_feedback(feedback_text):
    """Provide feedback and continue execution"""
    async for chunk in graph_with_checkpoint.astream(
        Command(resume=feedback_text), 
        {"configurable": {"thread_id": thread_id}},
        stream_mode="updates"
    ):
        print(chunk)
        print("\n")

        # Check if this chunk contains the final_report
        if isinstance(chunk, dict) and 'final_report' in chunk:
            print("🎉 Final report generated! 🎉")
            display_markdown(chunk['final_report'], "DeepSeek-R1 Report")
            return


async def run_research(capability_input: str):
    """Run the research process directly"""
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "planner_provider": "openai",
            "writer_provider": "openai",
            "search_api": "tavily",
            "planner_model": "gpt-4o-2024-08-06",  # Updated model name
            "writer_model": "gpt-4o-2024-08-06",   # Updated model name
            "method": "function_calling",
            "number_of_queries": 2,
            "max_search_depth": 2,
            "report_structure": DEFAULT_REPORT_STRUCTURE
        }
    }

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    try:
        async for chunk in graph.astream(
            {"capability": capability_input},  
            config,
            stream_mode="updates"
        ):
            if isinstance(chunk, dict):
                if 'final_report' in chunk:
                    try:
                        display_markdown(chunk['final_report'], f"# {capability_input} Report")
                    except NameError:  # If not in IPython environment
                        print(f"\n{capability_input} Report")
                        print("=" * 80)
                        print(chunk['final_report'])
                    return chunk['final_report']
                elif '__interrupt__' in chunk:
                    print("\nResearch Plan:")
                    print(chunk['__interrupt__'][0].value)
                    approval = input("\nDo you approve this plan? (yes/no): ").lower()
                    if approval == 'yes':
                        async for response in graph.astream(
                            Command(resume=True),
                            config,
                            stream_mode="updates"
                        ):
                            if isinstance(response, dict):
                                if 'final_report' in response:
                                    try:
                                        display_markdown(response['final_report'], f"# {capability_input} Report")
                                    except NameError:  # If not in IPython environment
                                        print(f"\n{capability_input} Report")
                                        print("=" * 80)
                                        print(response['final_report'])
                                    return response['final_report']
                                else:
                                    print(f"\nProgress: {str(response)}")
                    else:
                        feedback = input("\nPlease provide feedback for the plan: ")
                        async for response in graph.astream(
                            Command(resume=feedback),
                            config,
                            stream_mode="updates"
                        ):
                            if isinstance(response, dict):
                                if 'final_report' in response:
                                    try:
                                        display_markdown(response['final_report'], f"# {capability_input} Report")
                                    except NameError:  # If not in IPython environment
                                        print(f"\n{capability_input} Report")
                                        print("=" * 80)
                                        print(response['final_report'])
                                    return response['final_report']
                                else:
                                    print(f"\nProgress: {str(response)}")
                else:
                    print(f"\nProgress: {str(chunk)}")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Let user select the file
        file_path = select_file()
        print(f"Selected file: {file_path}")
        
        # Create retriever from selected file
        retriever = create_retriever_from_file(file_path)
        
        # Extract the capability
        capability_info = extract_capability(retriever)
        print(f"\nExtracted Capability: {capability_info['capability']}")
        print(f"\nSystem Description: {capability_info['system_description']}")
        
         # Run the research process
        print(f"\nStarting research for capability: {capability_info['capability']}")
        asyncio.run(run_research(capability_info['capability']))
        
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

