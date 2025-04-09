"""
MCP Tool Selector - Advanced user intent analysis for tool selection
"""

import logging
import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

log_file_path = os.path.join(logs_dir, 'mcp_selector.log')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mcp_selector')

# Load environment variables
load_dotenv()

class ToolRequirement(BaseModel):
    """Model for structured tool requirements"""
    tool_name: str = Field(..., description="The name of the tool (e.g., 'github', 'spotify')")
    importance: str = Field(..., description="Priority level: 'essential', 'important', or 'optional'")
    custom_features: List[str] = Field(default_factory=list, description="List of specific custom features requested")
    integration_points: List[str] = Field(default_factory=list, description="Systems this tool needs to connect with")
    authentication_requirements: List[str] = Field(default_factory=list, description="Required auth mechanisms")
    specific_endpoints: List[str] = Field(default_factory=list, description="Specific API endpoints mentioned")
    specific_use_cases: List[str] = Field(default_factory=list, description="Specific use cases or tasks this tool should accomplish")

class UserRequirements(BaseModel):
    """Complete model of user requirements for tool generation"""
    primary_tools: List[ToolRequirement] = Field(..., description="Main tools explicitly requested")
    secondary_tools: List[ToolRequirement] = Field(default_factory=list, description="Tools implicitly needed")
    customization_level: str = Field(..., description="Level of customization: 'standard', 'moderate', 'high'")
    special_instructions: List[str] = Field(default_factory=list, description="Special instructions for implementation")
    integration_pattern: str = Field(default="standalone", description="How tools should be integrated: 'standalone', 'interconnected'")
    workflow_description: str = Field(default="", description="Description of the workflow between tools if multiple")
    
async def extract_structured_requirements(query: str, openai_client: AsyncOpenAI) -> UserRequirements:
    """
    Extract structured tool requirements using a deep analysis of the user's query.
    
    Args:
        query: User's query/request
        openai_client: AsyncOpenAI client
        
    Returns:
        Structured UserRequirements object
    """
    try:
        logger.info(f"Performing deep requirement analysis for query")
        
        prompt = f"""
        Analyze the following user request to extract detailed requirements for MCP tool generation.
        Identify both explicit and implicit tool needs, special requirements, and customization details.
        
        USER REQUEST: "{query}"
        
        Provide your analysis in this JSON format:
        {{
            "primary_tools": [
                {{
                    "tool_name": "tool name",
                    "importance": "essential|important|optional",
                    "custom_features": ["feature 1", "feature 2"],
                    "integration_points": ["system 1", "system 2"],
                    "authentication_requirements": ["oauth", "api_key", etc],
                    "specific_endpoints": ["endpoint 1", "endpoint 2"],
                    "specific_use_cases": ["use case 1", "use case 2"]
                }}
            ],
            "secondary_tools": [similar format to primary_tools],
            "customization_level": "standard|moderate|high",
            "special_instructions": ["instruction 1", "instruction 2"],
            "integration_pattern": "standalone|interconnected",
            "workflow_description": "Description of how tools should work together in a workflow (if multiple tools)"
        }}
        
        Return ONLY valid JSON without explanations or comments.
        """
        
        response = await openai_client.chat.completions.create(
            model=os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse the JSON response
        try:
            requirements_dict = json.loads(result_text)
            requirements = UserRequirements.model_validate(requirements_dict)
            logger.info(f"Successfully extracted structured requirements. Primary tools: {[t.tool_name for t in requirements.primary_tools]}")
            return requirements
        except Exception as parse_error:
            logger.error(f"Error parsing requirements JSON: {str(parse_error)}")
            # Fall back to a simplified approach
            simple_tools = await get_required_tools(query, openai_client)
            
            # Create a basic requirements object
            primary_tools = [
                ToolRequirement(
                    tool_name=tool,
                    importance="essential",
                    custom_features=[],
                    integration_points=[],
                    authentication_requirements=[],
                    specific_endpoints=[],
                    specific_use_cases=[]
                ) for tool in simple_tools
            ]
            
            return UserRequirements(
                primary_tools=primary_tools,
                customization_level="standard",
                special_instructions=[],
                integration_pattern="standalone" if len(simple_tools) <= 1 else "interconnected",
                workflow_description=""
            )
    
    except Exception as e:
        logger.error(f"Error extracting structured requirements: {e}", exc_info=True)
        # Return minimal requirements structure
        return UserRequirements(
            primary_tools=[ToolRequirement(
                tool_name="general", 
                importance="essential",
                specific_use_cases=[]
            )],
            customization_level="standard",
            integration_pattern="standalone",
            workflow_description=""
        )

async def get_required_tools(query: str, openai_client: AsyncOpenAI) -> List[str]:
    """
    Directly ask OpenAI to identify which tools the user needs based on their query.
    More reliable than keyword matching for extracting user intent.
    
    Args:
        query: User's query/request
        openai_client: AsyncOpenAI client
        
    Returns:
        List of tool names the user wants to use
    """
    try:
        logger.info(f"Extracting tools directly from query using OpenAI")
        
        # Enhanced prompt that better detects implied tools
        prompt = f"""
        Based on the following user request, identify which specific external tools or APIs are needed.
        Consider both explicitly mentioned tools and those implied by the requested functionality.
        
        USER REQUEST: "{query}"
        
        Return ONLY a comma-separated list of the specific tools needed (e.g., "github, spotify"). 
        Known tools: github, spotify, youtube, twitter, slack, gmail, google_drive, discord, notion, trello, 
        asana, jira, instagram, linkedin, facebook, shopify, stripe, aws, serper, search, web_search.
        
        For search functionality, use "serper" or "search" as the tool name.
        For stock research, include both "serper" (for web search) and any financial APIs if mentioned.
        
        Do not include explanations or extra text - ONLY return the comma-separated list of tools.
        """
        
        response = await openai_client.chat.completions.create(
            model=os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1  # Low temperature for more deterministic answers
        )
        
        tools_text = response.choices[0].message.content.strip().lower()
        
        # Clean up the response
        tools_text = tools_text.replace(".", "").replace("and", ",")
        
        # Parse the comma-separated list
        tools = [tool.strip() for tool in tools_text.split(",") if tool.strip()]
        
        logger.info(f"OpenAI identified tools: {', '.join(tools)}")
        
        # Check if search capabilities are needed but not identified
        search_terms = ["research", "find", "search", "look up", "information about"]
        has_search_terms = any(term in query.lower() for term in search_terms)
        
        if has_search_terms and not any(search_tool in tools for search_tool in ["serper", "search", "web_search"]):
            logger.info("Search functionality implied but not detected, adding serper tool")
            tools.append("serper")
        
        return tools
        
    except Exception as e:
        logger.error(f"Error extracting tools from query: {e}")
        
        # Fallback to regex pattern matching for common tools
        logger.info("Falling back to pattern matching for tool detection")
        tool_patterns = {
            "github": r'github|git|repo|repository|code|pull request|commit',
            "spotify": r'spotify|music|playlist|song|track|artist',
            "serper": r'search|find|research|information|look up|discover',
            "youtube": r'youtube|video|watch|channel',
            "twitter": r'twitter|tweet|x\.com'
        }
        
        detected_tools = []
        for tool, pattern in tool_patterns.items():
            if re.search(pattern, query.lower()):
                detected_tools.append(tool)
                
        if detected_tools:
            logger.info(f"Regex pattern matching found tools: {', '.join(detected_tools)}")
            return detected_tools
        
        # If all else fails, at least provide search capability
        return ["serper"]

async def filter_tools_by_user_needs(tools: List[Dict[str, Any]], required_tools: List[str]) -> List[Dict[str, Any]]:
    """
    Filter a list of MCP tools to keep only those that match the user's required tools.
    
    Args:
        tools: List of tool dictionaries from the database
        required_tools: List of tool names the user specifically needs
        
    Returns:
        Filtered list of tools that match the required tools
    """
    if not required_tools:
        logger.info("No specific tools required, returning all tools")
        return tools
    
    filtered_tools = []
    for tool in tools:
        tool_purpose = tool.get('purpose', '').lower()
        
        # Check if this tool matches any of the required tools
        is_matching_tool = False
        for req_tool in required_tools:
            if req_tool in tool_purpose:
                is_matching_tool = True
                break
        
        if is_matching_tool:
            filtered_tools.append(tool)
    
    if filtered_tools:
        logger.info(f"Filtered from {len(tools)} to {len(filtered_tools)} tools matching user requirements")
        return filtered_tools
    else:
        logger.info(f"No tools matched the required tools, returning all tools")
        return tools

async def rank_tools_by_requirement_match(
    tools: List[Dict[str, Any]], 
    requirements: UserRequirements
) -> List[Dict[str, Any]]:
    """
    Rank tools based on how well they match the detailed user requirements.
    
    Args:
        tools: List of tool dictionaries from the database
        requirements: Structured user requirements
        
    Returns:
        Ranked list of tools with match score information
    """
    # Extract all required tool names from primary and secondary
    all_required_tool_names = [tool.tool_name for tool in requirements.primary_tools]
    all_required_tool_names.extend([tool.tool_name for tool in requirements.secondary_tools])
    
    # Map requirements by tool name for easy lookup
    req_by_name = {t.tool_name: t for t in requirements.primary_tools}
    req_by_name.update({t.tool_name: t for t in requirements.secondary_tools})
    
    # Add a match score to each tool
    scored_tools = []
    for tool in tools:
        score = 0
        matches = []
        tool_purpose = tool.get('purpose', '').lower()
        
        # Check tool name matches
        matching_tool_name = None
        for req_name in all_required_tool_names:
            if req_name in tool_purpose:
                matching_tool_name = req_name
                req_importance = req_by_name[req_name].importance
                # Score based on importance
                if req_importance == "essential":
                    score += 100
                elif req_importance == "important":
                    score += 50
                else:  # optional
                    score += 25
                matches.append(f"Tool matches {req_name} ({req_importance})")
                break
        
        # Skip tools that don't match any required tool
        if matching_tool_name is None:
            continue
            
        # Check for specific feature matches if we found a matching tool
        if matching_tool_name:
            req = req_by_name[matching_tool_name]
            tool_code = tool.get('tool_code', '')
            
            # Check for custom features
            for feature in req.custom_features:
                if feature.lower() in tool_code.lower():
                    score += 10
                    matches.append(f"Supports feature: {feature}")
            
            # Check for specific endpoints
            for endpoint in req.specific_endpoints:
                if endpoint.lower() in tool_code.lower():
                    score += 15
                    matches.append(f"Implements endpoint: {endpoint}")
            
            # Check for authentication methods
            for auth in req.authentication_requirements:
                if auth.lower() in tool_code.lower():
                    score += 10
                    matches.append(f"Supports auth: {auth}")
        
        # Add similarity score if available
        if 'similarity' in tool:
            similarity = float(tool.get('similarity', 0))
            # Scale similarity to 0-30 points
            similarity_score = similarity * 30
            score += similarity_score
            matches.append(f"Similarity score: {similarity:.2f}")
        
        # Add the scored tool
        tool_with_score = dict(tool)
        tool_with_score['match_score'] = score
        tool_with_score['match_reasons'] = matches
        scored_tools.append(tool_with_score)
    
    # Sort by match score (highest first)
    sorted_tools = sorted(scored_tools, key=lambda x: x.get('match_score', 0), reverse=True)
    
    if sorted_tools:
        logger.info(f"Ranked {len(sorted_tools)} tools by requirement match. Top score: {sorted_tools[0].get('match_score', 0)}")
    else:
        logger.info(f"No tools matched the specific requirements")
    
    return sorted_tools

# Add this function to help with tool selection for CrewAI format
async def get_crewai_tool_requirements(query: str, openai_client: AsyncOpenAI) -> Dict[str, Any]:
    """
    Extract detailed requirements for CrewAI tool generation.
    
    Args:
        query: User query requesting tool(s)
        openai_client: AsyncOpenAI client
        
    Returns:
        Dictionary with CrewAI-specific requirements
    """
    try:
        logger.info(f"Extracting CrewAI tool requirements from query")
        
        prompt = f"""
        Analyze this user request for CrewAI tool generation. Extract specific requirements.
        
        USER REQUEST: "{query}"
        
        Provide your analysis in this JSON format:
        {{
            "tools": [
                {{
                    "name": "clear tool name",
                    "functionality": "brief description of what it should do",
                    "authentication": "what auth is needed, if any",
                    "specific_api_endpoints": ["endpoint1", "endpoint2"],
                    "parameters": ["param1", "param2"]
                }}
            ],
            "integration_requirements": "any requirements about how tools work together"
        }}
        
        ONLY return valid JSON with no additional text.
        """
        
        response = await openai_client.chat.completions.create(
            model=os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content.strip()
        
        try:
            requirements = json.loads(result_text)
            logger.info(f"Extracted CrewAI requirements for {len(requirements.get('tools', []))} tools")
            return requirements
        except json.JSONDecodeError:
            logger.error("Failed to parse CrewAI requirements JSON")
            return {"tools": [{"name": "generic", "functionality": query}]}
            
    except Exception as e:
        logger.error(f"Error extracting CrewAI requirements: {str(e)}")
        return {"tools": [{"name": "generic", "functionality": query}]}

async def detect_keywords_in_query(query: str) -> List[str]:
    """
    Directly detect tool keywords in user query without relying on a model.
    
    Args:
        query: The user query string
        
    Returns:
        List of detected tool keywords
    """
    query = query.lower()
    detected_tools = []
    
    # Define keywords for each service
    keyword_lists = {
        "github": ["github", "git", "repository", "repo", "pull request", "pr", "issue", "commit", "branch", "merge"],
        "spotify": ["spotify", "music", "playlist", "song", "track", "artist", "album", "audio"],
        "youtube": ["youtube", "video", "channel", "stream", "youtube video"],
        "twitter": ["twitter", "tweet", "x.com", "tweets"],
        "slack": ["slack", "message", "channel", "slack message"],
        "gmail": ["gmail", "email", "mail", "inbox", "message"],
        "google_drive": ["google drive", "gdrive", "drive", "document", "sheet", "slides"],
        "discord": ["discord", "server", "channel"],
        "notion": ["notion", "page", "database", "notion page"],
        "trello": ["trello", "board", "card", "trello board"],
        "jira": ["jira", "ticket", "issue", "sprint"],
        "asana": ["asana", "task", "project", "assignee"],
        "instagram": ["instagram", "post", "story", "reel", "insta"],
        "linkedin": ["linkedin", "profile", "post", "connection"],
        "facebook": ["facebook", "post", "page", "fb", "meta"],
        "calendar": ["calendar", "event", "meeting", "schedule", "appointment", "gcal", "google calendar"],
        "shopify": ["shopify", "store", "product", "order", "customer"],
        "stripe": ["stripe", "payment", "invoice", "subscription", "charge"],
        "aws": ["aws", "amazon web services", "s3", "ec2", "lambda", "cloudwatch"],
        "openai": ["openai", "gpt", "dalle", "whisper", "chat gpt", "chatgpt", "gpt-4", "claude"],
        "zoom": ["zoom", "meeting", "call", "video call", "conference"],
        "hubspot": ["hubspot", "crm", "contact", "lead", "deal"],
        "salesforce": ["salesforce", "crm", "lead", "opportunity", "account"],
        "weather": ["weather", "forecast", "temperature", "humidity", "rain", "sun"],
        "maps": ["maps", "directions", "route", "navigation", "location", "google maps"],
        "news": ["news", "article", "headline", "current events"]
    }
    
    # Check for each keyword in the query
    for tool_name, keywords in keyword_lists.items():
        for keyword in keywords:
            if keyword in query:
                logger.info(f"MCP TOOL DETECTOR: Found keyword '{keyword}' for {tool_name} in query")
                if tool_name not in detected_tools:
                    detected_tools.append(tool_name)
                break  # Found one keyword for this tool, no need to check others
    
    if detected_tools:
        logger.info(f"MCP TOOL DETECTOR: Detected tools from keywords: {', '.join(detected_tools)}")
    else:
        logger.info("MCP TOOL DETECTOR: No tool keywords detected directly in query")
        
    return detected_tools

# Add this function as an alias for get_crewai_tool_requirements
async def extract_crewai_requirements(query: str, openai_client: AsyncOpenAI) -> Dict[str, Any]:
    """
    Extract detailed requirements for CrewAI tool generation.
    This is an alias for get_crewai_tool_requirements to maintain compatibility.
    
    Args:
        query: User query requesting tool(s)
        openai_client: AsyncOpenAI client
        
    Returns:
        Dictionary with CrewAI-specific requirements
    """
    return await get_crewai_tool_requirements(query, openai_client)

# Export the functions
__all__ = [
    'get_required_tools', 
    'filter_tools_by_user_needs',
    'extract_structured_requirements',
    'extract_crewai_requirements',  # Add the new function to __all__
    'rank_tools_by_requirement_match',
    'get_crewai_tool_requirements',
    'detect_keywords_in_query',
    'UserRequirements',
    'ToolRequirement'
] 