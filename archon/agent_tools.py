from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from supabase import Client
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var

embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'

async def get_embedding(text: str, embedding_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await embedding_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=1536
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def search_agent_templates_tool(supabase: Client, embedding_client: AsyncOpenAI, query: str, threshold: float = 0.4, limit: int = 3) -> Dict[str, Any]:
    """
    Search for agent templates using embedding similarity.
    
    Args:
        supabase: Supabase client
        embedding_client: OpenAI client for embeddings
        query: The search query describing the agent to build
        threshold: Similarity threshold (0.0 to 1.0)
        limit: Maximum number of results to return
        
    Returns:
        Dict containing similar agent templates with their code and metadata
    """
    try:
        # Get embedding for the query
        query_embedding = await get_embedding(query, embedding_client)
        
        # Search for similar templates using the search_agent_embeddings function
        result = supabase.rpc(
            'search_agent_embeddings',
            {
                'query_embedding': query_embedding,
                'similarity_threshold': threshold,
                'match_count': limit
            }
        ).execute()
        
        if not result.data:
            return {"templates": [], "message": "No similar templates found."}
        
        # Get full details for each template
        templates = []
        for match in result.data:
            template_id = match['id']
            full_template = supabase.table('agent_embeddings') \
                .select('*') \
                .eq('id', template_id) \
                .execute()
                
            if full_template.data:
                template = full_template.data[0]
                templates.append({
                    "id": template['id'],
                    "folder_name": template['folder_name'],
                    "purpose": template['purpose'],
                    "similarity": match['similarity'],
                    "agents_code": template['agents_code'],
                    "main_code": template['main_code'],
                    "models_code": template['models_code'],
                    "tools_code": template['tools_code'],
                    "mcp_json": template['mcp_json'],
                    "metadata": template['metadata']
                })
        
        return {
            "templates": templates,
            "message": f"Found {len(templates)} similar templates."
        }
        
    except Exception as e:
        print(f"Error searching agent templates: {e}")
        return {"templates": [], "message": f"Error searching agent templates: {str(e)}"}

async def fetch_template_by_id_tool(supabase: Client, template_id: int) -> Dict[str, Any]:
    """
    Fetch a specific agent template by ID.
    
    Args:
        supabase: Supabase client
        template_id: The ID of the template to fetch
        
    Returns:
        Dict containing the agent template with its code and metadata
    """
    try:
        result = supabase.table('agent_embeddings') \
            .select('*') \
            .eq('id', template_id) \
            .execute()
            
        if not result.data:
            return {"found": False, "message": f"No template found with ID: {template_id}"}
            
        template = result.data[0]
        return {
            "found": True,
            "template": {
                "id": template['id'],
                "folder_name": template['folder_name'],
                "purpose": template['purpose'],
                "agents_code": template['agents_code'],
                "main_code": template['main_code'],
                "models_code": template['models_code'],
                "tools_code": template['tools_code'],
                "mcp_json": template['mcp_json'],
                "metadata": template['metadata']
            }
        }
        
    except Exception as e:
        print(f"Error fetching template: {e}")
        return {"found": False, "message": f"Error fetching template: {str(e)}"}

async def retrieve_relevant_documentation_tool(supabase: Client, embedding_client: AsyncOpenAI, user_query: str) -> str:
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, embedding_client)
        
        # Query Supabase for relevant documents
        result = supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 4,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}" 

async def list_documentation_pages_tool(supabase: Client) -> List[str]:
    """
    Function to retrieve a list of all available Pydantic AI documentation pages.
    This is called by the list_documentation_pages tool and also externally
    to fetch documentation pages for the reasoner LLM.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

async def get_page_content_tool(supabase: Client, url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together but limit the characters in case the page is massive (there are a coule big ones)
        # This will be improved later so if the page is too big RAG will be performed on the page itself
        return "\n\n".join(formatted_content)[:20000]
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

def get_file_content_tool(file_path: str) -> str:
    """
    Retrieves the content of a specific file. Use this to get the contents of an example, tool, config for an MCP server

    Args:
        file_path: The path to the file
        
    Returns:
        The raw contents of the file
    """
    try:
        with open(file_path, "r") as file:
            file_contents = file.read()
        return file_contents
    except Exception as e:
        print(f"Error retrieving file contents: {e}")
        return f"Error retrieving file contents: {str(e)}"           