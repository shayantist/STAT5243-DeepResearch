# utils.py
import os
import asyncio
import requests
from typing import List, Optional, Dict, Any

from tavily import AsyncTavilyClient
from duckduckgo_search import DDGS 

from langsmith import traceable

def get_config_value(value):
    """
    Helper function to handle both string and enum cases of configuration values
    """
    return value if isinstance(value, str) else value.value

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
        "tavily": [],  # Tavily currently accepts no additional parameters
        "perplexity": [],  # Perplexity accepts no additional parameters
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source tokens.
 
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
    formatted_text = "Content from sources:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'='*80}\n"  # Clear section separator
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"{'-'*80}\n"  # Subsection separator
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
        formatted_text += f"{'='*80}\n\n" # End section separator
                
    return formatted_text.strip()


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
    tavily_async_client = AsyncTavilyClient()
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
async def duckduckgo_search(search_queries):
    """Perform searches using DuckDuckGo
    
    Args:
        search_queries (List[str]): List of search queries to process
        
    Returns:
        List[dict]: List of search results
    """
    async def process_single_query(query):
        # Execute synchronous search in the event loop's thread pool
        loop = asyncio.get_event_loop()
        
        def perform_search():
            results = []
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=5))
                
                # Format results
                for i, result in enumerate(ddg_results):
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('link', ''),
                        'content': result.get('body', ''),
                        'score': 1.0 - (i * 0.1),  # Simple scoring mechanism
                        'raw_content': result.get('body', '')
                    })
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
            
        return await loop.run_in_executor(None, perform_search)

    # Execute all queries concurrently
    tasks = [process_single_query(query) for query in search_queries]
    search_docs = await asyncio.gather(*tasks)
    
    return search_docs

@traceable
async def github_search(search_queries: list[str]):
    """Perform searches using GitHub API
    
    Args:
        search_queries (List[str]): List of search queries to process
        
    Returns:
        List[dict]: List of search results in consistent format with other search APIs
    """
    # Get GitHub token from environment
    github_token = os.environ.get("GITHUB_API_TOKEN")
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    async def process_single_query(query):
        results = []
        
        try:
            # Search repositories
            repo_url = "https://api.github.com/search/repositories"
            repo_params = {"q": query, "per_page": 3, "sort": "stars", "order": "desc"}
            
            # Search code - we'll scope this to popular AI/ML orgs by default
            code_url = "https://api.github.com/search/code"
            code_query = f"{query} org:langchain-ai OR org:openai OR org:microsoft OR org:google"
            code_params = {"q": code_query, "per_page": 3}
            
            # Search issues
            issues_url = "https://api.github.com/search/issues"
            issues_params = {"q": query, "per_page": 3, "sort": "updated", "order": "desc"}
            
            # Execute all searches concurrently
            loop = asyncio.get_event_loop()
            repo_resp, code_resp, issues_resp = await asyncio.gather(
                loop.run_in_executor(None, lambda: requests.get(repo_url, headers=headers, params=repo_params)),
                loop.run_in_executor(None, lambda: requests.get(code_url, headers=headers, params=code_params)),
                loop.run_in_executor(None, lambda: requests.get(issues_url, headers=headers, params=issues_params))
            )
            
            # Check rate limits
            def check_rate_limit(resp):
                if resp.status_code == 403 and "rate limit exceeded" in resp.text.lower():
                    reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                    wait_time = max(0, reset_time - time.time())
                    raise Exception(f"GitHub API rate limit exceeded. Try again in {wait_time:.0f} seconds.")
                resp.raise_for_status()
                return resp
            
            # Process repository results
            try:
                repo_data = check_rate_limit(repo_resp).json()
                for item in repo_data.get("items", [])[:3]:
                    results.append({
                        "title": f"Repository: {item.get('full_name', '')}",
                        "url": item.get("html_url", ""),
                        "content": item.get("description", "") or "No description",
                        "raw_content": f"Stars: {item.get('stargazers_count', 0)}\nLanguage: {item.get('language', 'Unknown')}",
                        "score": min(1.0, item.get("stargazers_count", 0)) / 1000
                    })
            except Exception:
                pass
            
            # Process code results
            try:
                code_data = check_rate_limit(code_resp).json()
                for item in code_data.get("items", [])[:3]:
                    results.append({
                        "title": f"Code: {item.get('name', '')}",
                        "url": item.get("html_url", ""),
                        "content": f"Path: {item.get('path', '')}",
                        "raw_content": f"Repository: {item.get('repository', {}).get('full_name', '')}",
                        "score": 0.7
                    })
            except Exception:
                pass
            
            # Process issue results
            try:
                issues_data = check_rate_limit(issues_resp).json()
                for item in issues_data.get("items", [])[:3]:
                    body = (item.get("body", "") or "No description")[:500]
                    results.append({
                        "title": f"Issue: {item.get('title', '')}",
                        "url": item.get("html_url", ""),
                        "content": body,
                        "raw_content": f"State: {item.get('state', 'open')}\nComments: {item.get('comments', 0)}",
                        "score": 0.5
                    })
            except Exception:
                pass
            
        except Exception:
            pass
            
        return {
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": [],
            "results": results[:5]
        }
    
    # Execute all queries with concurrency limit
    semaphore = asyncio.Semaphore(2)
    async def limited_search(query):
        async with semaphore:
            return await process_single_query(query)
    
    tasks = [limited_search(query) for query in search_queries]
    search_docs = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out any exceptions
    return [doc for doc in search_docs if not isinstance(doc, Exception)]

async def select_and_execute_search(search_api: str, query_list: list[str], params_to_pass: dict) -> str:
    """Select and execute the appropriate search API.
    
    Args:
        search_api: Name of the search API to use
        query_list: List of search queries to execute
        params_to_pass: Parameters to pass to the search API
        
    Returns:
        Formatted string containing search results
        
    Raises:
        ValueError: If an unsupported search API is specified
    """
    if search_api == "tavily":
        search_results = await tavily_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000, include_raw_content=False)
    elif search_api == "perplexity":
        search_results = perplexity_search(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "duckduckgo":
        search_results = await duckduckgo_search(query_list)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "googlesearch":
        search_results = await google_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "github":
        search_results = await github_search(query_list)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "hybrid":
        # Run both web and GitHub searches
        web_results = await tavily_search_async(query_list, **params_to_pass)
        github_results = await github_search(query_list)
        
        # Combine results
        combined_results = web_results + github_results
        
        return deduplicate_and_format_sources(combined_results, max_tokens_per_source=4000)
    else:
        raise ValueError(f"Unsupported search API: {search_api}")