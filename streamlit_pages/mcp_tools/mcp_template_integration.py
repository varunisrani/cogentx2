"""
MCP Template Integration Module

This module provides functions to integrate MCP templates from the template registry
into the MCP tool coder system. It focuses on using pre-made templates for agents.py,
tasks.py, and crew.py to speed up the creation of MCP-based CrewAI applications.
"""

import os
import json
import re
import logging
import asyncio
import math
import difflib
from typing import Dict, Any, List, Optional, Tuple
from openai import AsyncOpenAI
from supabase import Client
from pydantic_ai import RunContext
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger('mcp_templates')

# Template adapter class to store required information
@dataclass
class TemplateAdapter:
    template_id: str
    folder_name: str
    purpose: str
    similarity: float
    agents_code: str
    tasks_code: str
    crew_code: str
    metadata: Dict[str, Any]
    main_code: str = ""  # Add main_code field with default empty string
    run_agent_code: str = ""  # Add run_agent_code field with default empty string
    
    @property
    def agent_names(self) -> List[str]:
        """Get agent names from the metadata or extract them from code."""
        if self.metadata and 'agent_names' in self.metadata:
            return self.metadata['agent_names']
        else:
            # Extract agent names from code if not in metadata
            matches = re.findall(r'([a-zA-Z0-9_]+)\s*=\s*Agent\(', self.agents_code)
            return matches
            
    @property
    def tool_functions(self) -> List[str]:
        """Get tool functions from the metadata."""
        if self.metadata and 'tool_functions' in self.metadata:
            return self.metadata['tool_functions']
        return []

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def find_matching_mcp_template(
    user_query: str, 
    tools_data: Dict[str, Any],
    supabase: Client,
    openai_client: AsyncOpenAI,
    limit: int = 3,
    threshold: float = 0.45  # Lower threshold to capture more potential matches
) -> List[TemplateAdapter]:
    """
    Find matching MCP templates based on user query and tool data.
    
    This function performs a combined matching algorithm:
    1. Uses semantic search with embeddings
    2. Also compares the purpose field directly
    3. Considers the tools needed vs. tools available in templates
    
    Args:
        user_query: The user's query or request
        tools_data: Information about the tools being used
        supabase: Supabase client
        openai_client: AsyncOpenAI client
        limit: Maximum number of templates to return
        threshold: Minimum similarity threshold
        
    Returns:
        List of matching template adapters sorted by relevance
    """
    try:
        logger.info(f"Searching for MCP templates matching query: '{user_query[:100]}...'")
        
        # Extract tool types or names from tools_data
        tool_types = []
        tool_names = []
        
        if "tools" in tools_data and isinstance(tools_data["tools"], list):
            for tool in tools_data["tools"]:
                if 'type' in tool:
                    tool_types.append(tool['type'].lower())
                if 'name' in tool:
                    tool_names.append(tool['name'].lower())
                    # Also extract base name without "Tool" suffix for broader matching
                    if tool['name'].lower().endswith('tool'):
                        base_name = tool['name'].lower()[:-4]  # Remove 'tool' suffix
                        if base_name and base_name not in tool_names:
                            tool_names.append(base_name)
        
        # Extract tool class names from tool_class_names if available
        if "tool_class_names" in tools_data and isinstance(tools_data["tool_class_names"], list):
            for name in tools_data["tool_class_names"]:
                name_lower = name.lower()
                # Add tool name
                tool_names.append(name_lower)
                # Extract base name without "Tool", "MCPTool", etc.
                if "tool" in name_lower:
                    base_name = name_lower.replace("tool", "").replace("mcp", "").strip()
                    if base_name and base_name not in tool_names:
                        tool_names.append(base_name)
        
        # Check user query directly for known tool types
        if user_query:
            query_lower = user_query.lower()
            # Dictionary of known tools to check for
            known_tools = {
                "spotify": ["spotify", "music", "song", "playlist", "track", "artist", "album"],
                "github": ["github", "git", "repository", "repo", "pull request", "issue"],
                "search": ["search", "serper", "find", "google", "web search"],
                "youtube": ["youtube", "video", "channel"]
            }
            
            # Check each tool type
            for tool_type, keywords in known_tools.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        if tool_type not in tool_types:
                            tool_types.append(tool_type)
                            logger.info(f"Detected tool type '{tool_type}' from keyword '{keyword}' in query")
                        break
        
        # Log the detected tool types and names
        if tool_types:
            logger.info(f"Detected tool types: {', '.join(tool_types)}")
        if tool_names:
            logger.info(f"Detected tool names: {', '.join(tool_names)}")
            
        # Create a comprehensive combined search text
        purpose_parts = []
        
        # Add the user query prominently
        if user_query:
            purpose_parts.append(user_query)
            # Add it twice to increase its weight in the embedding
            purpose_parts.append(user_query)
        
        # Extract and add the general purpose of the tools
        general_purpose = ""
        if "purpose" in tools_data and tools_data.get("purpose"):
            general_purpose = tools_data.get("purpose")
            purpose_parts.append(general_purpose)
            # Add it twice to increase its weight
            purpose_parts.append(general_purpose)
        
        # Add tools purpose
        if "tools" in tools_data and isinstance(tools_data["tools"], list):
            for tool in tools_data["tools"]:
                if tool.get('purpose'):
                    purpose_parts.append(tool.get('purpose'))
        
        # Create combined text
        combined_search = " ".join(purpose_parts)
        
        # Extract key terms for more effective matching
        key_terms = set()
        
        # Add tool types and names
        for term in tool_types + tool_names:
            if term:
                key_terms.add(term)
        
        # Extract potential domain/task terms from user query and purpose
        domain_terms = extract_key_terms(user_query) | extract_key_terms(general_purpose)
        key_terms.update(domain_terms)
        
        # Add key terms to search text
        if key_terms:
            combined_search += f" {' '.join(key_terms)}"
            
        logger.info(f"Combined search text: '{combined_search[:100]}...'")
        logger.info(f"Key matching terms: {', '.join(key_terms)}")
        
        # Try direct purpose-based matching first
        purpose_matches = []
        
        # Get templates that might match the purpose semantically
        # First try direct purpose search for literal tool names/types
        for term in tool_names + tool_types:
            if not term or len(term) < 3:
                continue
                
            try:
                # Search in purpose field
                search_pattern = f"%{term}%"
                purpose_result = supabase.table("mcp_templates").select("*").ilike("purpose", search_pattern).execute()
                
                if purpose_result.data:
                    logger.info(f"Found {len(purpose_result.data)} purpose matches for '{term}'")
                    
                    for template in purpose_result.data:
                        try:
                            # Make sure all required fields are present
                            if not all(k in template for k in ['id', 'folder_name', 'purpose', 'agents_code', 'tasks_code', 'crew_code']):
                                logger.warning(f"Template {template.get('id', 'unknown')} is missing required fields, skipping")
                                continue
                                
                            # Boost similarity score based on how specific the match is
                            sim_score = 0.85  # Base score for purpose matches
                            
                            # Higher score for exact tool name matches
                            folder_lower = template['folder_name'].lower()
                            if any(t in folder_lower for t in tool_types) or any(n in folder_lower for n in tool_names):
                                sim_score = 0.95
                                logger.info(f"Boosted score for {template['folder_name']} - exact tool name/type match")
                            
                            adapter = TemplateAdapter(
                                template_id=template['id'],
                                folder_name=template['folder_name'],
                                purpose=template['purpose'],
                                similarity=sim_score,
                                agents_code=template['agents_code'],
                                tasks_code=template['tasks_code'],
                                crew_code=template['crew_code'],
                                metadata=template.get('metadata', {}),
                                main_code=template.get('main_code', ''),
                                run_agent_code=template.get('run_agent_code', '')
                            )
                            purpose_matches.append(adapter)
                        except Exception as e:
                            logger.warning(f"Error processing purpose match: {e}")
                            continue
            except Exception as e:
                logger.warning(f"Error during purpose search for term '{term}': {e}")

        # Try direct folder name matching next
        folder_matches = []
        for tool_type in tool_types + tool_names:
            # Look for pattern in folder_name
            if tool_type and len(tool_type) >= 3:
                try:
                    search_pattern = f"%{tool_type}%"
                    folder_result = supabase.table("mcp_templates").select("*").ilike("folder_name", search_pattern).execute()
                    
                    if folder_result.data:
                        logger.info(f"Found {len(folder_result.data)} folder matches for '{tool_type}'")
                        
                        for template in folder_result.data:
                            try:
                                # Make sure all required fields are present
                                if not all(k in template for k in ['id', 'folder_name', 'purpose', 'agents_code', 'tasks_code', 'crew_code']):
                                    logger.warning(f"Template {template.get('id', 'unknown')} is missing required fields, skipping")
                                    continue
                                
                                adapter = TemplateAdapter(
                                    template_id=template['id'],
                                    folder_name=template['folder_name'],
                                    purpose=template['purpose'],
                                    similarity=0.92,  # High similarity for folder name matches
                                    agents_code=template['agents_code'],
                                    tasks_code=template['tasks_code'],
                                    crew_code=template['crew_code'],
                                    metadata=template.get('metadata', {}),
                                    main_code=template.get('main_code', ''),
                                    run_agent_code=template.get('run_agent_code', '')
                                )
                                folder_matches.append(adapter)
                            except Exception as e:
                                logger.warning(f"Error processing folder match: {e}")
                                continue
                except Exception as e:
                    logger.warning(f"Error during folder name search for term '{tool_type}': {e}")
        
        # Combine and deduplicate direct matches
        direct_matches = []
        seen_ids = set()
        
        # Add folder matches first (higher priority)
        for match in folder_matches:
            if match.template_id not in seen_ids:
                direct_matches.append(match)
                seen_ids.add(match.template_id)
                
        # Add purpose matches next
        for match in purpose_matches:
            if match.template_id not in seen_ids:
                direct_matches.append(match)
                seen_ids.add(match.template_id)
        
        if direct_matches:
            logger.info(f"Using {len(direct_matches)} direct matches from tool name/type")
            # Sort by similarity score (highest first)
            direct_matches.sort(key=lambda x: x.similarity, reverse=True)
            return direct_matches[:limit]
        
        # Generate embedding for the combined search
        query_embedding = await get_embedding(combined_search, openai_client)
        
        # Perform vector search for templates
        try:
            # Ensure the threshold is a float
            match_threshold = float(threshold)
            
            res = supabase.rpc(
                'match_mcp_templates',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': match_threshold,
                    'match_count': int(limit * 3)  # Get more results for better filtering, ensure it's an integer
                }
            ).execute()
            
            if not res.data:
                logger.info("No matching templates found from embedding search.")
                return []
                
            logger.info(f"Found {len(res.data)} potential template matches from embedding search")
            
            # Get complete template data for each match
            template_matches = []
            for item in res.data:
                try:
                    template_id = item['id']
                    
                    # Ensure similarity is a float
                    try:
                        similarity = float(item['similarity'])
                    except (TypeError, ValueError):
                        # Default to a reasonable value if conversion fails
                        logger.warning(f"Could not convert similarity value to float: {item.get('similarity')}")
                        similarity = 0.5
                    
                    # Get the complete template data
                    template_data = supabase.table("mcp_templates").select("*").eq("id", template_id).execute()
                    
                    if not template_data.data:
                        continue
                        
                    template = template_data.data[0]
                    
                    # Boost similarity score based on matches
                    similarity_score = similarity
                    score_boosted = False
                    
                    # Check for key term matches in purpose (higher boost)
                    purpose_lower = template['purpose'].lower()
                    for term in key_terms:
                        if term and term in purpose_lower:
                            similarity_score += 0.15
                            logger.info(f"Boosted score for {template['folder_name']} - key term '{term}' found in purpose")
                            score_boosted = True
                            break
                    
                    # Boost for matching tool type in folder name 
                    folder_lower = template['folder_name'].lower()
                    for tool_type in tool_types + tool_names:
                        if tool_type and tool_type in folder_lower:
                            similarity_score += 0.12
                            logger.info(f"Boosted score for {template['folder_name']} - tool type '{tool_type}' match in name")
                            score_boosted = True
                            break
                    
                    # Calculate text similarity between user query and template purpose
                    if not score_boosted and user_query and template['purpose']:
                        text_sim = calculate_text_similarity(user_query, template['purpose'])
                        # Only boost if the text similarity is significant
                        if text_sim > 0.4:
                            boost = min(text_sim * 0.25, 0.2)  # Cap boost at 0.2
                            similarity_score += boost
                            logger.info(f"Boosted score for {template['folder_name']} by {boost:.2f} based on text similarity")
                    
                    # Cap similarity score at 0.99
                    similarity_score = min(similarity_score, 0.99)
                    
                    # Create a template adapter
                    adapter = TemplateAdapter(
                        template_id=template_id,
                        folder_name=template['folder_name'],
                        purpose=template['purpose'],
                        similarity=similarity_score,
                        agents_code=template['agents_code'],
                        tasks_code=template['tasks_code'],
                        crew_code=template['crew_code'],
                        metadata=template.get('metadata', {}),
                        main_code=template.get('main_code', ''),
                        run_agent_code=template.get('run_agent_code', '')
                    )
                    
                    template_matches.append(adapter)
                except Exception as e:
                    logger.warning(f"Error processing template match: {e}")
                    continue
            
            # Sort by similarity - Filter out NaN values and place them last
            valid_matches = [match for match in template_matches if not (isinstance(match.similarity, float) and math.isnan(match.similarity))]
            nan_matches = [match for match in template_matches if isinstance(match.similarity, float) and math.isnan(match.similarity)]
            
            # Sort valid matches by similarity score
            valid_matches.sort(key=lambda x: x.similarity, reverse=True)
            
            # Combine sorted valid matches with nan matches at the end
            sorted_matches = valid_matches + nan_matches
            
            # Take the top matches
            top_matches = sorted_matches[:limit]
            
            if top_matches:
                logger.info(f"Returning {len(top_matches)} top template matches")
                for i, match in enumerate(top_matches):
                    similarity_display = "NaN" if isinstance(match.similarity, float) and math.isnan(match.similarity) else f"{match.similarity:.3f}"
                    logger.info(f"Match {i+1}: {match.folder_name} (similarity: {similarity_display})")
                    logger.info(f"  Purpose: {match.purpose[:100]}...")
                return top_matches
                
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            # Continue to try text-based matching as fallback
        
        # Fallback: If no matches or error in vector search, try text-based matching
        logger.info("Attempting text-based matching as fallback")
        text_matches = []
        
        # Get all templates
        all_templates = supabase.table("mcp_templates").select("*").execute()
        
        if all_templates.data:
            logger.info(f"Checking {len(all_templates.data)} templates for text-based matching")
            
            # Calculate simple text match scores
            for template in all_templates.data:
                try:
                    # Calculate multiple similarity metrics
                    similarity_score = 0.0
                    
                    # 1. Check for direct key term matches
                    purpose_lower = template.get('purpose', '').lower() if template.get('purpose') else ''
                    folder_lower = template.get('folder_name', '').lower() if template.get('folder_name') else ''
                    
                    term_match_count = 0
                    for term in key_terms:
                        if term and term in purpose_lower:
                            term_match_count += 1
                        if term and term in folder_lower:
                            term_match_count += 0.5  # Half weight for folder matches
                    
                    if term_match_count > 0:
                        # Score based on percentage of key terms matched
                        term_score = min(0.6, (term_match_count / len(key_terms)) * 0.6) if key_terms else 0
                        similarity_score += term_score
                    
                    # 2. Compare user query directly with purpose
                    if user_query and template.get('purpose'):
                        text_sim = calculate_text_similarity(user_query, template['purpose'])
                        similarity_score += text_sim * 0.5  # Weight of 0.5 for text similarity
                    
                    # 3. Compare general purpose with template purpose
                    if general_purpose and template.get('purpose'):
                        purpose_sim = calculate_text_similarity(general_purpose, template['purpose'])
                        similarity_score += purpose_sim * 0.3  # Weight of 0.3 for purpose similarity
                    
                    # 4. Direct check for tool type in folder name (high priority)
                    for tool_type in tool_types:
                        if tool_type in folder_lower:
                            similarity_score += 0.3
                            logger.info(f"Boosted score for {template['folder_name']} - direct tool type match in folder name")
                    
                    # 5. Check for any tool name in folder
                    for tool_name in tool_names:
                        if tool_name in folder_lower:
                            similarity_score += 0.25
                            logger.info(f"Boosted score for {template['folder_name']} - tool name found in folder name")
                    
                    # Only consider if score is above minimum threshold
                    if similarity_score > threshold:
                        # Scale to 0-1 range
                        scaled_score = min(0.95, similarity_score)
                        
                        # Make sure all required fields are present
                        if not all(k in template for k in ['id', 'folder_name', 'purpose', 'agents_code', 'tasks_code', 'crew_code']):
                            logger.warning(f"Template {template.get('id', 'unknown')} is missing required fields, skipping")
                            continue
                        
                        adapter = TemplateAdapter(
                            template_id=template['id'],
                            folder_name=template['folder_name'],
                            purpose=template['purpose'],
                            similarity=scaled_score,
                            agents_code=template['agents_code'],
                            tasks_code=template['tasks_code'],
                            crew_code=template['crew_code'],
                            metadata=template.get('metadata', {}),
                            main_code=template.get('main_code', ''),
                            run_agent_code=template.get('run_agent_code', '')
                        )
                        text_matches.append(adapter)
                except Exception as e:
                    logger.warning(f"Error processing text-based match: {e}")
                    continue
            
            # Sort by similarity
            text_matches.sort(key=lambda x: x.similarity, reverse=True)
            
            # Take the top matches
            top_text_matches = text_matches[:limit]
            
            if top_text_matches:
                logger.info(f"Returning {len(top_text_matches)} top text-based matches")
                for i, match in enumerate(top_text_matches):
                    logger.info(f"Match {i+1}: {match.folder_name} (similarity: {match.similarity:.3f})")
                    logger.info(f"  Purpose: {match.purpose[:100]}...")
                return top_text_matches
        
        logger.info("No suitable template matches found using any method")
        return []
        
    except Exception as e:
        logger.error(f"Error finding matching MCP templates: {e}")
        return []

def extract_key_terms(text: str) -> set:
    """
    Extract key terms from text for better template matching.
    
    Args:
        text: Text to extract terms from
        
    Returns:
        Set of key terms
    """
    if not text:
        return set()
        
    # Convert to lowercase
    text = text.lower()
    
    # List of common task domains
    domains = {
        'search', 'web', 'github', 'google', 'api', 'youtube', 'spotify', 
        'twitter', 'news', 'finance', 'stock', 'weather', 'calendar', 
        'email', 'document', 'file', 'database', 'chat', 'message', 
        'audio', 'video', 'image', 'text', 'data', 'analysis', 'scraping',
        'serper', 'serpapi'
    }
    
    # Split text into words
    words = set(re.findall(r'\b\w+\b', text))
    
    # Find domain terms 
    found_domains = words.intersection(domains)
    
    # Add compound terms
    compound_terms = set()
    for i in range(len(text.split()) - 1):
        bigram = ' '.join(text.split()[i:i+2]).lower()
        if any(domain in bigram for domain in domains):
            compound_terms.add(bigram)
    
    return found_domains.union(compound_terms)

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings using simple algorithms.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
        
    # Convert to lowercase
    text1 = text1.lower()
    text2 = text2.lower()
    
    # Split into word sets
    words1 = set(re.findall(r'\b\w+\b', text1))
    words2 = set(re.findall(r'\b\w+\b', text2))
    
    # Remove common stopwords
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'from'}
    words1 = words1.difference(stopwords)
    words2 = words2.difference(stopwords)
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

async def adapt_agents_code(
    template: TemplateAdapter,
    tool_class_names: List[str],
    openai_client: AsyncOpenAI,
    model_name: str,
    user_query: str = ""
) -> str:
    """
    Adapt agents code from a template to use the specified tools.
    
    Args:
        template: The template adapter with template code
        tool_class_names: Names of the tool classes to use
        openai_client: AsyncOpenAI client
        model_name: Model name to use
        user_query: User's original query/requirements
        
    Returns:
        Adapted agents code
    """
    try:
        # Get the template code
        template_code = template.agents_code
        
        if not template_code or not template_code.strip():
            logger.error("Empty agents code in template")
            return ""
        
        # Create a more forceful prompt for agent adaptation
        prompt = f"""
        CRITICAL INSTRUCTION: You must COMPLETELY TRANSFORM this agents.py code.
        The current implementation is too generic and needs a complete overhaul.
        
        USER REQUIREMENTS:
        {user_query}
        
        TRANSFORMATION REQUIREMENTS (ALL MANDATORY):
        1. RENAME the main class to be TASK-SPECIFIC (e.g., "YouTubeTranscriptAgentFactory" for a YouTube transcript task)
        2. REMOVE all generic agent roles like "YouTube Transcript Expert" or "Language Specialist"
        3. CREATE agent roles that are DIRECTLY ALIGNED with the user's specific task
        4. ENSURE all agent names reflect their SPECIFIC FUNCTION in the requested task workflow
        5. REWRITE all agent backstories to include:
           - Specific technical expertise RELEVANT TO THE USER'S TASK
           - Years of domain experience IN THE EXACT TASK DOMAIN
           - Specialized certifications or training SPECIFIC TO THE TASK
           - Real-world accomplishments in the SAME TASK field
        6. MODIFY all agent goals to focus on concrete, measurable outcomes FOR THE SPECIFIC TASK
        7. CHANGE all method names to reflect their TASK-SPECIFIC purpose
        8. UPDATE tool assignments to create true experts IN THE REQUESTED TASK
        
        AVAILABLE TOOLS: {', '.join(tool_class_names)}
        
        ORIGINAL CODE:
        ```python
        {template_code}
        ```
        
        CRITICAL RULES:
        - You MUST change at least 80% of the code
        - NEVER use generic terms unrelated to the user's specific task
        - ALL agent names, roles, goals, and backstories MUST directly address the user's specific task
        - If user needs a "YouTube transcript analyzer", create agents like "youtube_transcript_extractor" and "transcript_insight_generator"
        - ALWAYS use specific terminology from the user's domain and task
        - NEVER create agents with purposes unrelated to the user's specific request
        - The agents.py MUST be generated according to the specific requirements provided by the user
        
        Return ONLY the transformed Python code with NO explanations.
        The code must be production-ready and properly indented.
        """
        
        # Call OpenAI API with higher temperature for more creative changes
        logger.info(f"Calling OpenAI API to adapt agents code with enhanced transformation")
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,  # Higher temperature for more creative changes
            max_tokens=4000
        )
        
        # Get the adapted code
        adapted_code = response.choices[0].message.content.strip()
        adapted_code = adapted_code.replace("```python", "").replace("```", "").strip()
        
        # Verify the changes are substantial
        change_percentage = calculate_change_percentage(template_code, adapted_code)
        logger.info(f"Initial change percentage for agents.py: {change_percentage:.2f}%")
        
        # If changes aren't substantial enough, try again with even more forceful prompt
        if change_percentage < 70:
            logger.info("Changes insufficient, attempting more aggressive transformation")
            retry_prompt = f"""
            EMERGENCY REWRITE REQUIRED - Previous adaptation was insufficient!
            
            You MUST create a COMPLETELY NEW agents.py file that is unrecognizable from the original.
            
            USER REQUIREMENTS:
            {user_query}
            
            ABSOLUTE REQUIREMENTS:
            1. DELETE every single generic agent role
            2. CREATE entirely new agent roles that DIRECTLY SERVE THE USER'S SPECIFIC TASK
            3. WRITE complex, detailed backstories for each agent that are HYPER-FOCUSED ON THE USER'S TASK
            4. INCLUDE specific technical skills and certifications RELEVANT TO COMPLETING THE USER'S TASK
            5. CHANGE every single method name to be TASK-SPECIFIC
            6. MODIFY all goals to include measurable outcomes FOR THE EXACT TASK REQUESTED
            7. REWRITE all tool assignments for maximum specialization IN THE TASK DOMAIN
            
            TOOLS TO USE: {', '.join(tool_class_names)}
            
            Previous code was too similar to:
            ```python
            {template_code}
            ```
            
            CRITICAL RULES:
            - CHANGE EVERYTHING - aim for 90%+ difference from original
            - ALL agents must have names that DIRECTLY REFLECT THEIR FUNCTION in the task workflow
            - If user requests a "stock market analyzer", create agents like "market_data_collector" and "stock_trend_analyzer"
            - EVERY agent MUST serve a specific purpose in completing the USER'S EXACT TASK
            - ALL terminology must match the USER'S TASK DOMAIN precisely
            - NEVER include agents with purposes unrelated to the user's specific request
            - The agents.py MUST be generated according to the specific requirements provided by the user
            
            Create completely new code that shares only the basic CrewAI structure.
            """
            
            retry_response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": retry_prompt}],
                temperature=1.0,  # Maximum creativity
                max_tokens=4000
            )
            
            adapted_code = retry_response.choices[0].message.content.strip()
            adapted_code = adapted_code.replace("```python", "").replace("```", "").strip()
            
            final_change_percentage = calculate_change_percentage(template_code, adapted_code)
            logger.info(f"Final change percentage for agents.py after retry: {final_change_percentage:.2f}%")
        
        logger.info(f"Successfully adapted agents code, {len(adapted_code)} characters")
        return adapted_code
        
    except Exception as e:
        logger.error(f"Error adapting agents code: {e}", exc_info=True)
        return ""

async def adapt_tasks_code(
    template: TemplateAdapter,
    agent_names: List[str],
    openai_client: AsyncOpenAI,
    model_name: str
) -> str:
    """
    Adapt the tasks code from a template to work with specific agents.
    
    Args:
        template: Template adapter containing the original tasks code
        agent_names: Names of agents from the adapted agents.py
        openai_client: AsyncOpenAI client
        model_name: Model to use for adaptation
        
    Returns:
        Adapted tasks.py code
    """
    try:
        logger.info(f"Adapting tasks code from template: {template.folder_name}")
        
        # Create a prompt for adaptation
        prompt = f"""
        You need to adapt the tasks.py code from a template to work with new agent names.
        
        ORIGINAL TASKS CODE FROM TEMPLATE:
        ```python
        {template.tasks_code}
        ```
        
        NEW AGENT NAMES TO USE: {', '.join(agent_names)}
        TEMPLATE'S ORIGINAL AGENT NAMES: {', '.join(template.agent_names)}
        
        Please adapt the code to:
        1. Import the correct agent names from agents.py: {', '.join(agent_names)}
        2. Assign tasks to the appropriate agents
        3. Keep the overall task structure and purposes
        4. Make sure task descriptions match the capabilities of the agents
        
        Return ONLY the complete Python code for the adapted tasks.py file.
        """
        
        # Use the model to adapt the code
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500,
            temperature=0.5
        )
        
        adapted_code = response.choices[0].message.content
        
        # Clean up the code if wrapped in markdown
        if "```python" in adapted_code:
            adapted_code = adapted_code.split("```python", 1)[1]
            if "```" in adapted_code:
                adapted_code = adapted_code.split("```", 1)[0]
        elif "```" in adapted_code:
            adapted_code = adapted_code.split("```", 1)[1]
            if "```" in adapted_code:
                adapted_code = adapted_code.split("```", 1)[0]
        
        logger.info(f"Successfully adapted tasks code, {len(adapted_code)} characters")
        return adapted_code.strip()
        
    except Exception as e:
        logger.error(f"Error adapting tasks code: {e}")
        # Return the original code if adaptation fails
        return template.tasks_code

async def adapt_crew_code(
    template: TemplateAdapter,
    agent_names: List[str],
    task_names: List[str],
    openai_client: AsyncOpenAI,
    model_name: str
) -> str:
    """
    Adapt the crew code from a template to work with specific agents and tasks.
    
    Args:
        template: Template adapter containing the original crew code
        agent_names: Names of agents from the adapted agents.py
        task_names: Names of tasks from the adapted tasks.py
        openai_client: AsyncOpenAI client
        model_name: Model to use for adaptation
        
    Returns:
        Adapted crew.py code
    """
    try:
        logger.info(f"Adapting crew code from template: {template.folder_name}")
        
        # Extract template's original agent and task names
        agent_pattern = re.findall(r'from\s+agents\s+import\s+([^#\n]+)', template.crew_code)
        template_agent_names = []
        if agent_pattern:
            template_agent_names = [name.strip() for name in agent_pattern[0].split(',')]
            
        task_pattern = re.findall(r'from\s+tasks\s+import\s+([^#\n]+)', template.crew_code)
        template_task_names = []
        if task_pattern:
            template_task_names = [name.strip() for name in task_pattern[0].split(',')]
        
        # Create a prompt for adaptation
        prompt = f"""
        You need to adapt the crew.py code from a template to work with new agent and task names.
        
        ORIGINAL CREW CODE FROM TEMPLATE:
        ```python
        {template.crew_code}
        ```
        
        NEW AGENT NAMES TO USE: {', '.join(agent_names)}
        NEW TASK NAMES TO USE: {', '.join(task_names)}
        
        TEMPLATE'S ORIGINAL AGENT NAMES: {', '.join(template_agent_names)}
        TEMPLATE'S ORIGINAL TASK NAMES: {', '.join(template_task_names)}
        
        Please adapt the code to:
        1. Import the correct agent names from agents.py: {', '.join(agent_names)}
        2. Import the correct task names from tasks.py: {', '.join(task_names)}
        3. Use these agents and tasks in the crew configuration
        4. Keep the overall crew structure and workflow
        5. Make sure the crew name and description match the purpose
        
        Return ONLY the complete Python code for the adapted crew.py file.
        """
        
        # Use the model to adapt the code
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3500,
            temperature=0.5
        )
        
        adapted_code = response.choices[0].message.content
        
        # Clean up the code if wrapped in markdown
        if "```python" in adapted_code:
            adapted_code = adapted_code.split("```python", 1)[1]
            if "```" in adapted_code:
                adapted_code = adapted_code.split("```", 1)[0]
        elif "```" in adapted_code:
            adapted_code = adapted_code.split("```", 1)[1]
            if "```" in adapted_code:
                adapted_code = adapted_code.split("```", 1)[0]
        
        logger.info(f"Successfully adapted crew code, {len(adapted_code)} characters")
        return adapted_code.strip()
        
    except Exception as e:
        logger.error(f"Error adapting crew code: {e}")
        # Return the original code if adaptation fails
        return template.crew_code

async def adapt_tools_code(
    template: TemplateAdapter,
    original_tools_code: str,
    tool_class_names: List[str],
    agents_code: str,
    openai_client: AsyncOpenAI,
    model_name: str
) -> str:
    """
    Adapt the tools.py file based on template and agent requirements.
    
    Args:
        template: The template adapter with metadata
        original_tools_code: The original tools.py code
        tool_class_names: List of tool class names
        agents_code: The adapted agents.py code
        openai_client: AsyncOpenAI client
        model_name: Model to use for adaptation
        
    Returns:
        Adapted tools.py code
    """
    try:
        logger.info(f"Adapting tools.py to match template requirements")
        
        # Get tool functions from template if available
        required_functions = template.tool_functions
        
        # If the template has metadata about required tool methods, include that
        tool_methods = []
        if template.metadata and "tool_methods" in template.metadata:
            tool_methods = template.metadata["tool_methods"]
            logger.info(f"Template requires these tool methods: {', '.join(tool_methods)}")
        
        # Create a comprehensive prompt for adapting the tools
        prompt = f"""
        Adapt the original tools.py code to work with the agents in the adapted agents.py file.
        
        Original tools.py:
        ```python
        {original_tools_code}
        ```
        
        Agents.py that will use these tools:
        ```python
        {agents_code}
        ```
        
        Your task is to update the tools.py file to:
        1. Ensure all tools needed by the agents are properly implemented
        2. Add any missing methods or functionality required by the agents
        3. Keep the original tool class names: {', '.join(tool_class_names)}
        4. Do NOT remove any existing functionality - only add or enhance
        
        IMPORTANT: Pay special attention to these common errors and fix them preemptively:
        1. Class naming inconsistency: If a tool class is named 'XYZTool', make sure to also create 'XYZMCPTool' as an alias class for backward compatibility
        2. Missing tool suffix: All tool classes should end with 'Tool' suffix (e.g., 'YouTube' should be 'YouTubeTool')
        3. Inconsistent method names: Standardize method names across the codebase (e.g., use 'get_transcript' instead of 'extract_transcript')
        4. Tool class naming convention: Always provide an alias class with 'MCPTool' suffix that inherits from the main tool class
        
        For YouTube tools specifically, ensure both 'YouTubeTranscriptTool' and 'YouTubeTranscriptMCPTool' classes exist, with one inheriting from the other.
        """
        
        # Add specific requirements if they exist in the template
        if required_functions:
            prompt += f"\n\nThe template requires these specific tool functions:\n"
            for func in required_functions:
                prompt += f"- {func}\n"
                
        if tool_methods:
            prompt += f"\n\nEnsure these methods are implemented in the tool classes:\n"
            for method in tool_methods:
                prompt += f"- {method}\n"
        
        prompt += """
        
        Return ONLY the complete updated tools.py code with no additional explanations.
        """
        
        # Use the model to adapt the code
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.2
        )
        
        adapted_code = response.choices[0].message.content
        
        # Clean up the code if wrapped in markdown
        if "```python" in adapted_code:
            adapted_code = adapted_code.split("```python", 1)[1]
            if "```" in adapted_code:
                adapted_code = adapted_code.split("```", 1)[0]
        elif "```" in adapted_code:
            adapted_code = adapted_code.split("```", 1)[1]
            if "```" in adapted_code:
                adapted_code = adapted_code.split("```", 1)[0]
        
        # Import the validate_tools_py_content function to perform validation
        try:
            from .mcp_tools.mcp_tool_coder import validate_tools_py_content
            
            # Validate and fix common errors in the tools.py content
            fixed_content, applied_fixes = validate_tools_py_content(adapted_code)
            
            # If fixes were applied, log them
            if applied_fixes:
                logger.info(f"Applied {len(applied_fixes)} fixes to tools.py content")
                for fix in applied_fixes:
                    logger.info(f"  - {fix['message']}")
                
                # Use the fixed content instead
                adapted_code = fixed_content
        except ImportError:
            logger.warning("Could not import validate_tools_py_content, skipping validation")
        
        logger.info(f"Successfully adapted tools.py, {len(adapted_code)} characters")
        return adapted_code.strip()
        
    except Exception as e:
        logger.error(f"Error adapting tools.py: {e}")
        # Return the original code if adaptation fails
        return original_tools_code

async def extract_names_from_code(code: str, pattern: str) -> List[str]:
    """
    Extract names from code using a regex pattern.
    
    Args:
        code: The code to extract names from
        pattern: Regex pattern to use
        
    Returns:
        List of extracted names
    """
    try:
        matches = re.findall(pattern, code)
        return [m for m in matches if m]
    except Exception as e:
        logger.error(f"Error extracting names from code: {e}")
        return []

def create_executable_scripts(output_dir: str) -> None:
    """
    Create shell scripts to make Python files executable.
    
    Args:
        output_dir: Directory where Python files are located
    """
    try:
        # Create chmod script for Unix/Mac systems
        chmod_script = os.path.join(output_dir, "make_executable.sh")
        with open(chmod_script, "w") as f:
            f.write("""#!/bin/bash
# Make Python scripts executable
chmod +x *.py
echo "Made Python scripts executable"
""")
        # Make the script itself executable
        try:
            os.chmod(chmod_script, 0o755)
            logger.info(f"Created and made executable: {chmod_script}")
        except Exception as e:
            logger.warning(f"Could not make script executable: {e}")
            
        # Create README.md with instructions if it doesn't exist
        readme_file = os.path.join(output_dir, "README.md")
        if not os.path.exists(readme_file):
            with open(readme_file, "w") as f:
                f.write(f"""# CrewAI Application

## Scripts
This directory contains several Python scripts:

- **main.py**: Main entry point for the CrewAI application with full functionality
- **run_agent.py**: Simple script to run a single agent directly
- **agents.py**: Defines the agents used in the application
- **tasks.py**: Defines the tasks assigned to agents
- **crew.py**: Orchestrates the agents and tasks
- **tools.py**: Contains tools used by the agents

## Running the application
1. Make the scripts executable (on Unix/Mac):
   ```
   ./make_executable.sh
   ```
   
2. Run the main application:
   ```
   ./main.py
   ```
   
3. To run a single agent directly:
   ```
   ./run_agent.py "your query here"
   ```

## Requirements
- Python 3.9 or higher
- Required packages: crewai, langchain, and any other dependencies
""")
            logger.info(f"Created README.md with usage instructions")
            
    except Exception as e:
        logger.error(f"Error creating executable scripts: {e}")

async def generate_from_template(
    user_query: str,
    tools_data: Dict[str, Any],
    tool_class_names: List[str],
    output_dir: str,
    supabase: Client,
    openai_client: AsyncOpenAI,
    model_name: str
) -> Dict[str, str]:
    """
    Generate a complete CrewAI project from templates.
    
    Args:
        user_query: User's query
        tools_data: Information about the tools being used
        tool_class_names: Names of tool classes from tools.py
        output_dir: Output directory for the generated files
        supabase: Supabase client
        openai_client: AsyncOpenAI client
        model_name: Model name to use for adaptation
        
    Returns:
        Dict with generated file contents
    """
    try:
        # Find matching templates
        templates = await find_matching_mcp_template(
            user_query, 
            tools_data, 
            supabase, 
            openai_client
        )
        
        if not templates:
            logger.warning("No matching templates found")
            return {}
            
        # Use the best matching template
        template = templates[0]
        logger.info(f"Using template: {template.folder_name} (similarity: {template.similarity:.3f})")
        logger.info(f"Template purpose: {template.purpose[:100]}...")
        
        # Adapt the agents code
        logger.info("Adapting agents code...")
        adapted_agents = await adapt_agents_code(
            template,
            tool_class_names,
            openai_client,
            model_name,
            user_query
        )
        
        # Save the adapted agents code to file
        agents_file = os.path.join(output_dir, "agents.py")
        with open(agents_file, "w") as f:
            f.write(adapted_agents)
        logger.info(f"Saved adapted agents.py to {agents_file}")
        
        # No need to extract agent names - go directly to modifying the code for user requirements
        
        # Prepare results
        results = {"agents.py": adapted_agents}
        
        # Now adapt the tasks code
        tasks_code = await direct_requirements_adaptation(
            user_query,
            {
                "tasks.py": template.tasks_code
            },
            tools_data,
            tool_class_names,
            openai_client,
            model_name
        )
        
        # Save the tasks code
        tasks_file = os.path.join(output_dir, "tasks.py")
        with open(tasks_file, "w") as f:
            f.write(tasks_code["tasks.py"])
        logger.info(f"Saved adapted tasks.py to {tasks_file}")
        
        results["tasks.py"] = tasks_code["tasks.py"]
        
        # Now adapt the crew code
        crew_code = await direct_requirements_adaptation(
            user_query,
            {
                "crew.py": template.crew_code
            },
            tools_data,
            tool_class_names,
            openai_client,
            model_name
        )
        
        # Save the crew code
        crew_file = os.path.join(output_dir, "crew.py")
        with open(crew_file, "w") as f:
            f.write(crew_code["crew.py"])
        logger.info(f"Saved adapted crew.py to {crew_file}")
        
        results["crew.py"] = crew_code["crew.py"]
        
        # Generate main.py if main_code is available in the template
        if hasattr(template, 'main_code') and template.main_code:
            logger.info("Adapting main.py code...")
            main_code = await direct_requirements_adaptation(
                user_query,
                {
                    "main.py": template.main_code
                },
                tools_data,
                tool_class_names,
                openai_client,
                model_name
            )
            
            # Save the main.py file
            main_file = os.path.join(output_dir, "main.py")
            with open(main_file, "w") as f:
                # Ensure it starts with shebang
                content = main_code
                if not isinstance(content, str):
                    logger.warning(f"main_code is not a string, it's a {type(content)}. Attempting to convert.")
                    if isinstance(content, dict) and "main.py" in content:
                        content = content["main.py"]
                    else:
                        content = str(content)
                        
                if not content.startswith("#!/usr/bin/env python"):
                    content = "#!/usr/bin/env python3\n" + content
                f.write(content)
                
            # Make it executable
            try:
                os.chmod(main_file, 0o755)
                logger.info(f"Made main.py executable")
            except Exception as e:
                logger.warning(f"Could not make main.py executable: {e}")
                
            logger.info(f"Saved default main.py to {main_file}")
            
            results["main.py"] = content
        else:
            # Generate a default main.py file if not available in template
            logger.info("Generating default main.py file...")
            main_code = await generate_default_main_py(
                user_query,
                tool_class_names,
                openai_client,
                model_name
            )
                
            # Save the default main.py file
            main_file = os.path.join(output_dir, "main.py")
            with open(main_file, "w") as f:
                # Ensure it starts with shebang
                content = main_code
                if not isinstance(content, str):
                    logger.warning(f"main_code is not a string, it's a {type(content)}. Attempting to convert.")
                    if isinstance(content, dict) and "main.py" in content:
                        content = content["main.py"]
                    else:
                        content = str(content)
                        
                if not content.startswith("#!/usr/bin/env python"):
                    content = "#!/usr/bin/env python3\n" + content
                f.write(content)
                
            # Make it executable
            try:
                os.chmod(main_file, 0o755)
                logger.info(f"Made main.py executable")
            except Exception as e:
                logger.warning(f"Could not make main.py executable: {e}")
                
            logger.info(f"Saved default main.py to {main_file}")
            
            results["main.py"] = content
        
        # Generate run_agent.py file
        if hasattr(template, 'run_agent_code') and template.run_agent_code:
            logger.info("Adapting run_agent.py code...")
            run_agent_code = await direct_requirements_adaptation(
                user_query,
                {
                    "run_agent.py": template.run_agent_code
                },
                tools_data,
                tool_class_names,
                openai_client,
                model_name
            )
            
            # Save the run_agent.py file
            run_agent_file = os.path.join(output_dir, "run_agent.py")
            with open(run_agent_file, "w") as f:
                # Ensure it starts with shebang
                content = run_agent_code["run_agent.py"]
                if not isinstance(content, str):
                    logger.warning(f"run_agent_code is not a string, it's a {type(content)}. Attempting to convert.")
                    if isinstance(content, dict) and "run_agent.py" in content:
                        content = content["run_agent.py"]
                    else:
                        content = str(content)
                        
                if not content.startswith("#!/usr/bin/env python"):
                    content = "#!/usr/bin/env python3\n" + content
                f.write(content)
            
            # Make it executable
            try:
                os.chmod(run_agent_file, 0o755)
                logger.info(f"Made run_agent.py executable")
            except Exception as e:
                logger.warning(f"Could not make run_agent.py executable: {e}")
                
            logger.info(f"Saved adapted run_agent.py to {run_agent_file}")
            
            results["run_agent.py"] = content
        else:
            # Generate a default run_agent.py file if not available in template
            logger.info("Generating default run_agent.py file...")
            run_agent_code = await generate_run_agent_py(
                user_query,
                tool_class_names,
                openai_client,
                model_name,
                results.get("main.py")  # Pass main.py content for reference
            )
                
            # Save the default run_agent.py file
            run_agent_file = os.path.join(output_dir, "run_agent.py")
            with open(run_agent_file, "w") as f:
                # Ensure it starts with shebang
                content = run_agent_code
                if not isinstance(content, str):
                    logger.warning(f"run_agent_code is not a string, it's a {type(content)}. Attempting to convert.")
                    if isinstance(content, dict) and "run_agent.py" in content:
                        content = content["run_agent.py"]
                    else:
                        content = str(content)
                        
                if not content.startswith("#!/usr/bin/env python"):
                    content = "#!/usr/bin/env python3\n" + content
                f.write(content)
            
            # Make it executable
            try:
                os.chmod(run_agent_file, 0o755)
                logger.info(f"Made run_agent.py executable")
            except Exception as e:
                logger.warning(f"Could not make run_agent.py executable: {e}")
                
            logger.info(f"Saved default run_agent.py to {run_agent_file}")
            
            results["run_agent.py"] = content
            
        # Create executable scripts for main.py and run_agent.py
        create_executable_scripts(output_dir)
            
        # Perform cross-file validation and consistency check
        logger.info("Performing final cross-file validation and consistency check...")
        
        # Check tool class imports across all files
        changes_made = False
        
        # 1. Extract tool class names from tools.py or use provided tool_class_names
        actual_tool_classes = tool_class_names
        
        # 2. Extract agent class names from agents.py
        agent_classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*\(', results["agents.py"])
        agent_factory_methods = re.findall(r'def\s+create_([a-zA-Z0-9_]+)', results["agents.py"])
        
        logger.info(f"Found agent classes: {agent_classes}")
        logger.info(f"Found agent factory methods: {agent_factory_methods}")
        
        # 3. Verify all imports in agents.py use correct tool class names
        agent_imports = re.findall(r'from\s+tools\s+import\s+([^#\n]+)', results["agents.py"])
        if agent_imports:
            agent_tool_imports = [t.strip() for t in agent_imports[0].split(',')]
            for imported_tool in agent_tool_imports:
                if imported_tool and imported_tool not in actual_tool_classes:
                    logger.warning(f"Found inconsistent tool import in agents.py: {imported_tool}")
                    # Try to find closest match
                    for correct_tool in actual_tool_classes:
                        if imported_tool.startswith(correct_tool[:-4]) or correct_tool.startswith(imported_tool[:-4]):
                            new_agents_code = results["agents.py"].replace(imported_tool, correct_tool)
                            results["agents.py"] = new_agents_code
                            with open(agents_file, "w") as f:
                                f.write(new_agents_code)
                            logger.info(f"Fixed tool import in agents.py: {imported_tool} -> {correct_tool}")
                            changes_made = True
                            break
        
        # 4. Verify imports in crew.py match agent class names
        crew_imports = re.findall(r'from\s+agents\s+import\s+([^#\n]+)', results["crew.py"])
        if crew_imports and agent_classes:
            crew_agent_imports = [a.strip() for a in crew_imports[0].split(',')]
            logger.info(f"Crew imports: {crew_agent_imports}")
            
            # Find all agent class instantiations in crew.py
            agent_instantiations = re.findall(r'self\.(\w+)\s*=\s*(\w+)\(', results["crew.py"])
            logger.info(f"Agent instantiations in crew.py: {agent_instantiations}")
            
            # Add all non-standard agent classes to a list of classes that need fixing
            missing_or_incorrect_agents = []
            for var_name, class_name in agent_instantiations:
                if class_name not in agent_classes and class_name not in ["Agent", "Crew", "Task"]:
                    missing_or_incorrect_agents.append((var_name, class_name))
            
            # Also check direct class references in the code
            class_references = re.findall(r'=\s*(\w+)\([\s\n]*youtube_tool\s*=', results["crew.py"])
            for class_name in class_references:
                if class_name not in agent_classes and class_name not in crew_agent_imports:
                    missing_or_incorrect_agents.append(("referenced", class_name))
            
            logger.info(f"Missing or incorrect agent classes in crew.py: {missing_or_incorrect_agents}")
            
            # For each missing or incorrect agent, find the best match from agents.py
            for var_name, class_name in missing_or_incorrect_agents:
                best_match = None
                best_score = float('inf')
                
                for agent_class in agent_classes:
                    if 'Factory' in agent_class:
                        # Skip factory classes
                        continue
                    
                    # Calculate similarity score between incorrect class and actual agent class
                    score = levenshtein_distance(class_name.lower(), agent_class.lower())
                    
                    # If the class name appears in a factory method, give it a better score
                    for factory_method in agent_factory_methods:
                        method_name = f"create_{factory_method}"
                        if class_name.lower() in method_name.lower() or method_name.lower() in class_name.lower():
                            score -= 3  # Boost the score
                    
                    if score < best_score:
                        best_score = score
                        best_match = agent_class
                
                # If found a match, replace all occurrences of the incorrect class
                if best_match:
                    logger.info(f"Best match for {class_name} is {best_match} (score: {best_score})")
                    
                    # Replace the class name in imports
                    if class_name in crew_agent_imports:
                        import_pattern = re.compile(r'from\s+agents\s+import\s+([^#\n]+)')
                        imports = import_pattern.search(results["crew.py"]).group(1)
                        new_imports = imports.replace(class_name, best_match)
                        results["crew.py"] = import_pattern.sub(f'from agents import {new_imports}', results["crew.py"])
                    else:
                        # Add the import if it doesn't exist
                        if "from agents import" in results["crew.py"]:
                            results["crew.py"] = re.sub(
                                r'from\s+agents\s+import\s+([^#\n]+)',
                                f'from agents import \\1, {best_match}',
                                results["crew.py"]
                            )
                        else:
                            # Add a new import line
                            results["crew.py"] = re.sub(
                                r'(from\s+\w+\s+import\s+[^#\n]+\n)',
                                f'\\1from agents import {best_match}\n',
                                results["crew.py"],
                                count=1
                            )
                    
                    # Replace instantiations in code
                    results["crew.py"] = re.sub(
                        r'(\w+\s*=\s*)' + re.escape(class_name) + r'(\s*\()',
                        f'\\1{best_match}\\2',
                        results["crew.py"]
                    )
                    
                    # Replace direct references
                    results["crew.py"] = re.sub(
                        r'(=\s*)' + re.escape(class_name) + r'(\s*\([\s\n]*youtube_tool\s*=)',
                        f'\\1{best_match}\\2',
                        results["crew.py"]
                    )
                    
                    # Also check self references
                    results["crew.py"] = re.sub(
                        r'(self\.\w+\s*=\s*)' + re.escape(class_name) + r'(\s*\()',
                        f'\\1{best_match}\\2',
                        results["crew.py"]
                    )
                    
                    # Write the changes back to the file
                    with open(crew_file, "w") as f:
                        f.write(results["crew.py"])
                    
                    logger.info(f"Fixed agent class references in crew.py: {class_name} -> {best_match}")
                    changes_made = True
            
            # Check factory method usage
            for factory_name in agent_factory_methods:
                factory_pattern = f'create_{factory_name}'
                if factory_pattern in results["crew.py"]:
                    # Make sure the correct factory class is imported
                    for agent_class in agent_classes:
                        if 'Factory' in agent_class:
                            if agent_class not in crew_agent_imports:
                                # Add the factory class to imports
                                if "from agents import" in results["crew.py"]:
                                    results["crew.py"] = re.sub(
                                        r'from\s+agents\s+import\s+([^#\n]+)',
                                        f'from agents import \\1, {agent_class}',
                                        results["crew.py"]
                                    )
                                else:
                                    # Add a new import line
                                    results["crew.py"] = re.sub(
                                        r'(from\s+\w+\s+import\s+[^#\n]+\n)',
                                        f'\\1from agents import {agent_class}\n',
                                        results["crew.py"],
                                        count=1
                                    )
                                
                                # Write the changes back to the file
                                with open(crew_file, "w") as f:
                                    f.write(results["crew.py"])
                                
                                logger.info(f"Added factory class import to crew.py: {agent_class}")
                                changes_made = True
                                
                            # Make sure factory methods are called with the correct class
                            factory_calls = re.findall(r'(\w+)\.create_' + re.escape(factory_name), results["crew.py"])
                            for call_class in factory_calls:
                                if call_class != agent_class:
                                    results["crew.py"] = re.sub(
                                        re.escape(call_class) + r'\.create_' + re.escape(factory_name),
                                        f'{agent_class}.create_{factory_name}',
                                        results["crew.py"]
                                    )
                                    
                                    # Write the changes back to the file
                                    with open(crew_file, "w") as f:
                                        f.write(results["crew.py"])
                                    
                                    logger.info(f"Fixed factory method call in crew.py: {call_class}.create_{factory_name} -> {agent_class}.create_{factory_name}")
                                    changes_made = True

            # Check if there are missing imports for agent classes
            missing_imports = []
            for agent_class in agent_classes:
                # Skip factory classes since we handle them separately
                if 'Factory' in agent_class:
                    continue
                    
                if agent_class not in crew_agent_imports and agent_class in results["crew.py"]:
                    missing_imports.append(agent_class)
            
            if missing_imports:
                if "from agents import" in results["crew.py"]:
                    import_pattern = re.compile(r'from\s+agents\s+import\s+([^#\n]+)')
                    imports = import_pattern.search(results["crew.py"]).group(1)
                    new_imports = imports + ", " + ", ".join(missing_imports)
                    results["crew.py"] = import_pattern.sub(f'from agents import {new_imports}', results["crew.py"])
                else:
                    # Add a new import line
                    import_line = f'from agents import {", ".join(missing_imports)}\n'
                    # Add after other imports
                    if re.search(r'from\s+\w+\s+import', results["crew.py"]):
                        results["crew.py"] = re.sub(
                            r'(from\s+\w+\s+import\s+[^#\n]+\n)',
                            f'\\1{import_line}',
                            results["crew.py"],
                            count=1
                        )
                    else:
                        # Add at the top after any docstrings and comments
                        results["crew.py"] = re.sub(
                            r'^((?:#.*\n|""".*?"""\n)*)',
                            f'\\1{import_line}',
                            results["crew.py"],
                            flags=re.DOTALL
                        )
                
                # Write the changes back to the file
                with open(crew_file, "w") as f:
                    f.write(results["crew.py"])
                
                logger.info(f"Added missing agent imports to crew.py: {missing_imports}")
                changes_made = True
        
        # Verify method calls in tasks.py match factory methods in agents.py
        if "tasks.py" in results and agent_factory_methods:
            for factory_method in agent_factory_methods:
                method_pattern = f'create_{factory_method}'
                if method_pattern not in results["tasks.py"]:
                    # Check for similar method names that might be incorrect
                    similar_methods = re.findall(r'create_([a-zA-Z0-9_]+)', results["tasks.py"])
                    for similar in similar_methods:
                        if similar != factory_method and (
                            similar.lower() in factory_method.lower() or 
                            factory_method.lower() in similar.lower() or
                            levenshtein_distance(similar.lower(), factory_method.lower()) <= 3
                        ):
                            new_tasks_code = results["tasks.py"].replace(f'create_{similar}', f'create_{factory_method}')
                            results["tasks.py"] = new_tasks_code
                            with open(tasks_file, "w") as f:
                                f.write(new_tasks_code)
                            logger.info(f"Fixed method call in tasks.py: create_{similar} -> create_{factory_method}")
                            changes_made = True
                            break
        
        # Check for any tool class references in main.py and fix them
        if "main.py" in results:
            main_imports = re.findall(r'from\s+tools\s+import\s+([^#\n]+)', results["main.py"])
            if main_imports:
                main_tool_imports = [t.strip() for t in main_imports[0].split(',')]
                for imported_tool in main_tool_imports:
                    if imported_tool and imported_tool not in actual_tool_classes:
                        logger.warning(f"Found inconsistent tool import in main.py: {imported_tool}")
                        # Try to find closest match
                        for correct_tool in actual_tool_classes:
                            if imported_tool.startswith(correct_tool[:-4]) or correct_tool.startswith(imported_tool[:-4]):
                                new_main_code = results["main.py"].replace(imported_tool, correct_tool)
                                results["main.py"] = new_main_code
                                with open(main_file, "w") as f:
                                    f.write(new_main_code)
                                logger.info(f"Fixed tool import in main.py: {imported_tool} -> {correct_tool}")
                                changes_made = True
                                break
        
        # Check for any tool class references in run_agent.py and fix them
        if "run_agent.py" in results:
            run_agent_imports = re.findall(r'from\s+tools\s+import\s+([^#\n]+)', results["run_agent.py"])
            if run_agent_imports:
                run_agent_tool_imports = [t.strip() for t in run_agent_imports[0].split(',')]
                for imported_tool in run_agent_tool_imports:
                    if imported_tool and imported_tool not in actual_tool_classes:
                        logger.warning(f"Found inconsistent tool import in run_agent.py: {imported_tool}")
                        # Try to find closest match
                        for correct_tool in actual_tool_classes:
                            if imported_tool.startswith(correct_tool[:-4]) or correct_tool.startswith(imported_tool[:-4]):
                                new_run_agent_code = results["run_agent.py"].replace(imported_tool, correct_tool)
                                results["run_agent.py"] = new_run_agent_code
                                with open(run_agent_file, "w") as f:
                                    f.write(new_run_agent_code)
                                logger.info(f"Fixed tool import in run_agent.py: {imported_tool} -> {correct_tool}")
                                changes_made = True
                                break
                                
        # Check for agent references in run_agent.py and ensure they match agents.py
        if "run_agent.py" in results and agent_classes:
            for agent_class in agent_classes:
                # Skip factory classes when checking agent references
                if 'Factory' in agent_class:
                    continue
                    
                # Look for direct references to agent classes
                if agent_class in results["run_agent.py"] and f'from agents import {agent_class}' not in results["run_agent.py"]:
                    logger.warning(f"Agent class {agent_class} is used but not properly imported in run_agent.py")
                    # Try to add or fix the import
                    if 'from agents import' in results["run_agent.py"]:
                        # Extend existing import
                        original_import = re.search(r'from agents import\s+([^\n]+)', results["run_agent.py"]).group(0)
                        new_import = original_import + f', {agent_class}'
                        new_run_agent_code = results["run_agent.py"].replace(original_import, new_import)
                    else:
                        # Add new import at the top of the file
                        import_lines = re.findall(r'^import[^\n]+', results["run_agent.py"], re.MULTILINE)
                        if import_lines:
                            last_import = import_lines[-1]
                            new_run_agent_code = results["run_agent.py"].replace(
                                last_import, 
                                f"{last_import}\nfrom agents import {agent_class}"
                            )
                        else:
                            # If no imports found, add at the top after shebang or docstring
                            new_run_agent_code = re.sub(
                                r'(^#!.*?\n|^""".*?"""\n)',
                                r'\1\nfrom agents import {agent_class}\n',
                                results["run_agent.py"],
                                flags=re.DOTALL
                            )
                    results["run_agent.py"] = new_run_agent_code
                    with open(run_agent_file, "w") as f:
                        f.write(new_run_agent_code)
                    logger.info(f"Added import for agent class {agent_class} to run_agent.py")
                    changes_made = True
        
        if changes_made:
            logger.info("Cross-file consistency fixes applied successfully")
        else:
            logger.info("No cross-file consistency issues detected")
            
        return results
        
    except Exception as e:
        logger.error(f"Error generating from template: {e}")
        return {}

def calculate_change_percentage(original: str, modified: str) -> float:
    """
    Calculate the percentage change between two strings.
    
    Args:
        original: Original string
        modified: Modified string
        
    Returns:
        Percentage of change (0-100)
    """
    # Remove whitespace and normalize for comparison
    original_norm = re.sub(r'\s+', ' ', original).strip()
    modified_norm = re.sub(r'\s+', ' ', modified).strip()
    
    # Use difflib to compare the strings
    matcher = difflib.SequenceMatcher(None, original_norm, modified_norm)
    similarity = matcher.ratio()
    
    # Convert similarity to change percentage
    change_percentage = (1 - similarity) * 100
    
    return change_percentage

async def extract_customization_directives(user_query: str, openai_client: AsyncOpenAI) -> Tuple[int, List[str]]:
    """
    Extract specific customization directives from the user query.
    
    Args:
        user_query: The user's query or request
        openai_client: AsyncOpenAI client
        
    Returns:
        Tuple of (agent_count, domain_terms)
    """
    try:
        prompt = f"""
        Analyze this user request for a CrewAI tool and extract:
        
        1. The number of agents explicitly or implicitly required (default to multiple if unclear)
        2. A list of 5-10 domain-specific terms that should be used in the code
        
        USER REQUEST: "{user_query}"
        
        Return your analysis in this JSON format:
        {{
            "agent_count": <number of agents>,
            "domain_terms": [<list of domain-specific terms>]
        }}
        
        IMPORTANT: For agent_count, use a specific number (1, 2, 3, etc). 
        For domain_terms, provide terms that are highly specific to the domain, not generic technology terms.
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        agent_count = result.get("agent_count", 3)  # Default to 3 if not specified
        domain_terms = result.get("domain_terms", ["specialized", "customized", "domain-specific"])
        
        return agent_count, domain_terms
        
    except Exception as e:
        logger.error(f"Error extracting customization directives: {str(e)}")
        return 3, ["specialized", "customized", "domain-specific"]  # Default values

async def generate_default_main_py(
    user_query: str,
    tool_class_names: List[str],
    openai_client: AsyncOpenAI,
    model_name: str
) -> str:
    """
    Generate a default main.py file with enhanced CLI and error handling.
    
    Args:
        user_query: The user's query or request
        tool_class_names: List of tool class names
        openai_client: AsyncOpenAI client
        model_name: Model to use for generation
        
    Returns:
        Generated main.py content as a string
    """
    try:
        tools_text = ", ".join([f"`{cls}`" for cls in tool_class_names]) if tool_class_names else "specified tools"
        
        prompt = f"""
        Create a professional-grade main.py file for a CrewAI project that precisely implements this user's request:
        
        USER REQUEST: "{user_query}"
        
        The project uses these tool classes: {tools_text}
        
        YOUR MAIN.PY FILE MUST INCLUDE:
        
        1. COMPREHENSIVE COMMAND-LINE INTERFACE:
           - Use argparse to create a robust CLI with at least these arguments:
             * --verbose/-v: Control logging verbosity
             * --output/-o: Specify output file path
             * --format/-f: Specify output format (json, text, etc.)
             * --config/-c: Path to optional config file
             * Task-specific parameters relevant to this specific application
        
        2. ROBUST ERROR HANDLING:
           - Implement try-except blocks for ALL potential failure points
           - Handle missing files, import errors, authentication failures, API errors
           - Provide clear, informative error messages that help users fix problems
           - Implement proper exit codes for different error types
        
        3. PROFESSIONAL LOGGING:
           - Set up logging with configurable levels (DEBUG, INFO, WARNING, ERROR)
           - Include timestamps and log formatting
           - Log to both console and file with different verbosity levels
           - Add context-rich log messages that aid debugging
        
        4. USER-FRIENDLY OUTPUT:
           - Format results in a clean, readable way
           - Support multiple output formats (text, JSON)
           - Include summary statistics when appropriate
           - Show progress indicators for long-running operations
        
        5. CLEAN ARCHITECTURE:
           - Separate main functionality into well-organized functions
           - Include a properly structured __main__ block
           - Import crew correctly (from the crew.py file)
           - Keep configuration separate from execution logic
        
        YOUR IMPLEMENTATION MUST BE SPECIFICALLY TAILORED to this exact user request with domain-specific parameters, error messages, and output formatting.
        
        Return ONLY the complete Python code for main.py with no explanations or markdown formatting.
        """
        
        # Call OpenAI API to generate main.py
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2500
        )
        
        # Get the generated code
        main_py_content = response.choices[0].message.content.strip()
        
        # Clean up the code - remove any markdown formatting
        main_py_content = main_py_content.replace("```python", "").replace("```", "").strip()
        
        logger.info("Successfully generated default main.py with enhanced CLI and error handling")
        
        return main_py_content
    
    except Exception as e:
        logger.error(f"Error generating default main.py: {e}", exc_info=True)
        return f"""#!/usr/bin/env python3
# Default main.py file for CrewAI project
# Error during generation: {str(e)}

import argparse
import logging
import sys
from crew import get_crew

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("crew_main")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run the CrewAI application")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize and run crew
        crew = get_crew()
        result = crew.kickoff()
        print(result)
        return 0
    except Exception as e:
        logger.error(f"Error running crew: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
""" 

async def generate_run_agent_py(
    user_query: str,
    tool_class_names: List[str],
    openai_client: AsyncOpenAI,
    model_name: str,
    main_py_content: str = None
) -> str:
    """
    Generate a run_agent.py file for direct agent interaction.
    
    Args:
        user_query: The user's query or request
        tool_class_names: List of tool class names
        openai_client: AsyncOpenAI client
        model_name: Model to use for generation
        main_py_content: Optional content of main.py to use as reference
        
    Returns:
        String with generated run_agent.py content
    """
    try:
        tools_text = ", ".join([f"`{cls}`" for cls in tool_class_names]) if tool_class_names else "specified tools"
        
        # If main.py content is provided, we can reference it for consistency
        main_py_reference = ""
        if main_py_content:
            main_py_reference = f"""
            MAIN.PY REFERENCE (USE SAME IMPORTS & STYLE):
            ```python
            {main_py_content[:1000]}  # First 1000 chars for reference
            ```
            """
        
        prompt = f"""
        Create a specialized run_agent.py file for a CrewAI project. This file will provide a simple way to run a single agent directly.
        
        USER REQUEST: "{user_query}"
        
        The project uses these tool classes: {tools_text}
        
        {main_py_reference}
        
        YOUR RUN_AGENT.PY FILE MUST INCLUDE:
        
        1. SIMPLIFIED COMMAND-LINE INTERFACE:
           - Accept input text directly as an argument or via stdin if not provided
           - Support basic flags for configuration:
             * --verbose/-v: Control logging verbosity
             * --agent/-a: Specify which agent to run (if multiple are available)
             * --output/-o: Specify output file path
           
        2. ROBUST ERROR HANDLING:
           - Implement try-except blocks for potential failure points
           - Handle authentication failures and API errors gracefully
           - Provide helpful error messages
           
        3. DIRECT AGENT INVOCATION:
           - Import and create the agent directly from agents.py
           - Apply any necessary tool configurations
           - Print the agent's response to the terminal
           
        4. USER-FRIENDLY INTERACTION:
           - Support both direct command-line input and interactive mode
           - Format output in a readable way
           - Show progress when processing takes time
        
        YOUR IMPLEMENTATION MUST BE:
        1. SIMPLE - fewer than 100 lines of code
        2. FOCUSED specifically on running a single agent with minimal setup
        3. CONSISTENT with the project's imports and style
        4. DOMAIN-SPECIFIC - tailored to this exact user request with appropriate terminology
        
        Return ONLY the complete Python code for run_agent.py with no explanations or markdown formatting.
        """
        
        # Call OpenAI API to generate run_agent.py
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2500
        )
        
        # Get the generated code
        run_agent_py_content = response.choices[0].message.content.strip()
        
        # Clean up the code - remove any markdown formatting
        run_agent_py_content = run_agent_py_content.replace("```python", "").replace("```", "").strip()
        
        logger.info("Successfully generated run_agent.py with simplified agent execution interface")
        
        return run_agent_py_content
    
    except Exception as e:
        logger.error(f"Error generating run_agent.py: {e}", exc_info=True)
        
        # Create a basic fallback version
        fallback_content = f"""#!/usr/bin/env python3
# Simple agent runner for CrewAI project
# Error during generation: {str(e)}

import argparse
import logging
import sys
from agents import create_agent  # Assumes a create_agent function exists

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("agent_runner")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run a single agent directly")
    parser.add_argument("input", nargs="?", help="Input text for the agent")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-a", "--agent", default="primary", help="Agent to run (if multiple available)")
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get input from stdin if not provided as argument
    user_input = args.input
    if not user_input:
        print("Enter input for the agent (Ctrl+D to finish):")
        user_input = sys.stdin.read().strip()
    
    try:
        # Create and run the agent
        agent = create_agent(args.agent)
        result = agent.run(user_input)
        print(result)
        return 0
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
        
        return fallback_content

async def direct_requirements_adaptation(
    user_query: str,
    template_files: Dict[str, str],
    tools_data: Dict[str, Any],
    tool_class_names: List[str],
    openai_client: AsyncOpenAI,
    model_name: str = "gpt-4o"
) -> Dict[str, str]:
    """
    Directly adapt template files to the user's requirements using a single prompt.
    
    This approach uses a single, comprehensive prompt to adapt all template files
    to the user's requirements in one go.
    
    Args:
        user_query: The user's query or request
        template_files: Dictionary of template files to adapt
        tools_data: Data about the tools being integrated
        tool_class_names: List of tool class names
        openai_client: AsyncOpenAI client
        model_name: Model to use for adaptation
        
    Returns:
        Dictionary of adapted files
    """
    try:
        # Extract key information from tools data
        tools = tools_data.get("tools", [])
        tool_purposes = [tool.get("purpose", "") for tool in tools]
        detected_tool_types = tools_data.get("detected_tool_types", [])
        
        # Extract agent count and domain-specific terminology
        agent_count, domain_terms = await extract_customization_directives(user_query, openai_client)
        
        # Create a comprehensive adaptation prompt for all files
        prompt = f"""
        # USER REQUEST
        {user_query}

        # CUSTOMIZATION DIRECTIVE
        You MUST create DEEPLY CUSTOMIZED code for a CrewAI application based on the user's request.
        The current templates are TOO GENERIC and need DRAMATIC changes. 
        NUMBER OF AGENTS REQUIRED: {agent_count} (You MUST delete any unnecessary agents)
        DOMAIN TERMS TO USE: {', '.join(domain_terms)}
        
        Required changes (MANDATORY):
        1. REMOVE any agents, tasks, or code that don't directly support the specific request
        2. COMPLETELY REWRITE all agent roles, goals, and backstories with domain-specific language
        3. RESTRUCTURE the workflow to match the exact requirements
        4. RENAME all classes, methods, and variables to use domain-specific terminology
        5. ADD specialized functionality specifically requested by the user
        
        # RULES (CRITICAL)
        - Your adaptation must change AT MINIMUM 70% of the code
        - Generic terms like "YouTube Transcript Expert" MUST be replaced with specific terms like "Financial Video Content Analyzer"
        - If the user requests ONE agent, you MUST delete all other agent implementations
        - ALL descriptions must explicitly reference specific domain concepts
        - DO NOT preserve code just because it's in the template - RUTHLESSLY remove anything not needed
        - The final code must show SUBSTANTIAL CREATIVE DIFFERENCE from the template
        - KEEP CONSISTENT NAMING between files - tool classes must have the SAME NAME in all files
        
        # AGENTS.PY CUSTOMIZATION REQUIREMENTS
        - REMOVE any agent factory methods that don't match the user's needs
        - COMPLETELY REWRITE each agent's role, goal, and backstory to be HIGHLY SPECIFIC to the domain
        - CHANGE the agent class names to reflect their specific functions
        - MODIFY agent tool assignments to match exactly what's needed
        - ADD domain-specific knowledge and capabilities to each agent
        
        # TASKS.PY CUSTOMIZATION REQUIREMENTS  
        - REWRITE task descriptions to directly reference domain terminology
        - RESTRUCTURE tasks to reflect user's specific workflow needs
        - REMOVE generic task code and replace with domain-specific implementations
        - MODIFY task expected outputs to be relevant to the specific use case
        - ADD specific steps that reference domain concepts
        
        # CREW.PY CUSTOMIZATION REQUIREMENTS
        - REVISE the crew implementation to only include necessary agents
        - RESTRUCTURE workflow to match specific requirements
        - REWRITE logging and output handling to focus on domain-specific information
        - SIMPLIFY if unnecessary complexity exists
        - ADD specialized handling for the specific use case
        
        # TOOLS.PY CUSTOMIZATION REQUIREMENTS
        - PRESERVE core functionality but RENAME methods to use domain terminology
        - MODIFY parameter handling to be more specific to use case
        - ADD any specialized functionality needed for the domain
        - ADJUST error handling for domain-specific cases
        
        # MAIN.PY CUSTOMIZATION REQUIREMENTS
        - REWRITE command line interface to use domain-specific terminology
        - CUSTOMIZE output formatting for the specific use case
        - SIMPLIFY if only basic functionality is needed
        - ADD domain-specific configuration options

        # AVAILABLE FILES
        {json.dumps({k: f"{len(v)} characters" for k, v in template_files.items()}, indent=2)}

        # AVAILABLE TOOL CLASSES
        {', '.join(tool_class_names)}

        # TOOLS/SERVICES MENTIONED IN REQUEST
        {', '.join(detected_tool_types) if detected_tool_types else "None specifically mentioned"}

        # TOOL PURPOSES
        {chr(10).join([f"- {purpose}" for purpose in tool_purposes])}
        """
        
        # Process each file individually with the comprehensive customization prompt
        customized_files = {}
        
        for file_name, file_content in template_files.items():
            # Special handling for agents.py file to ensure deeper customization
            additional_instructions = ""
            if file_name == "agents.py":
                additional_instructions = f"""
                SPECIAL INSTRUCTIONS FOR AGENTS.PY:
                This file REQUIRES EXTRAORDINARY CUSTOMIZATION. The standard adaptation has been failing.
                
                YOU MUST:
                1. RENAME the main class to reflect the domain (e.g., "FinancialVideoAgentFactory", "StockAnalysisAgentFactory")
                2. REPLACE every agent role title with a domain-specific title (e.g., "Financial Data Analyst" instead of "YouTube Transcript Expert")
                3. COMPLETELY REWRITE all agent backstories to include specific domain knowledge and experience
                4. INCLUDE explicit references to at least 3 of these domain terms in each backstory: {', '.join(domain_terms)}
                5. MODIFY the goals to reflect specific outcomes relevant to the domain
                6. CHANGE method names to reflect their specific domain purpose
                7. ADJUST the multi_agent_team method to create specialized domain experts
                
                WARNING: Generic agent definitions will be REJECTED. Each agent must be a specialized domain expert.
                
                IMPORTANT: When you import tool classes, you MUST use the EXACT class names provided in the available tool classes:
                {', '.join(tool_class_names)}
                """
            
            file_prompt = f"""
            {prompt}
            
            # CURRENT FILE TO CUSTOMIZE: {file_name}
            
            This file requires DRAMATIC customization to meet the user's needs. The current template is FAR TOO GENERIC.
            {additional_instructions}
            
            YOUR TASK:
            1. Completely transform this code to fulfill the user's specific requirements
            2. Apply ALL customization requirements listed above
            3. Use domain-specific terminology throughout
            4. Make substantial structural changes where needed
            5. Create code that feels purpose-built for THIS specific use case, not generic
            6. ENSURE ALL TOOL CLASS NAMES are EXACTLY what was provided: {', '.join(tool_class_names)}
            
            # CURRENT FILE CONTENT
            ```python
            {file_content}
            ```
            
            Respond with ONLY the fully customized version of this file, with NO explanations.
            All code should be production-ready, properly indented, and fully functional.
            """
            
            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": file_prompt}],
                temperature=0.7,  # Higher temperature for more creative variations
                max_tokens=4000
            )
            
            # Extract the adapted file content
            adapted_content = response.choices[0].message.content
            
            # Clean up any markdown code blocks
            adapted_content = adapted_content.replace("```python", "").replace("```", "").strip()
            
            # Validate that meaningful changes were made
            change_percentage = calculate_change_percentage(file_content, adapted_content)
            logger.info(f"Change percentage for {file_name}: {change_percentage:.2f}%")
            
            # Extra validation for agents.py to ensure it's properly customized
            if file_name == "agents.py":
                # Check if common generic terms are still present
                generic_terms = ["YouTube Transcript Expert", "Transcript Specialist", 
                                "Language Specialist", "Educational Content Analyst",
                                "YouTube", "Transcript", "Video Content", "Expert", "Specialist"]
                
                # Count occurrences of the specific domain terms
                domain_term_count = sum(1 for term in domain_terms if term.lower() in adapted_content.lower())
                generic_term_count = sum(1 for term in generic_terms if term in adapted_content)
                
                # If too many generic terms remain or too few domain terms were added
                if generic_term_count > 1 or domain_term_count < min(3, len(domain_terms)):
                    logger.warning(f"agents.py customization insufficient - generic terms: {generic_term_count}, domain terms: {domain_term_count}")
                    change_percentage = 20  # Force a retry
            
            if file_name == "agents.py" and (change_percentage < 40 or generic_term_count > 0):
                # Special handling for agents.py - attempt a third time with even more forceful prompt if needed
                final_retry_prompt = f"""
                CRITICAL FAILURE: Your customization is still too generic.
                
                You MUST create a COMPLETELY NEW VERSION of this agents.py file.
                
                REQUIREMENTS:
                - RENAME the class to "{domain_terms[0] if domain_terms else 'Domain'}AgentFactory"
                - DELETE all occurrences of "YouTube Transcript Expert", "Transcript Specialist", etc.
                - CREATE new agent roles like "{domain_terms[0] if domain_terms else 'Domain'} Analyst", "{domain_terms[1] if len(domain_terms) > 1 else 'Specialized'} Researcher"
                - WRITE completely new backstories that mention: {', '.join(domain_terms[:3] if domain_terms else ['domain knowledge'])}
                - CHANGE all goals to focus specifically on {domain_terms[0] if domain_terms else 'the specific domain'}
                
                # USER REQUEST
                {user_query}
                
                # DOMAIN TERMS TO USE (MANDATORY)
                {', '.join(domain_terms)}
                
                # TOOL CLASSES - MUST USE EXACTLY THESE CLASS NAMES:
                {', '.join(tool_class_names)}
                
                DO NOT PRESERVE ANY YOUTUBE-SPECIFIC TERMINOLOGY.
                
                Write a completely new file from scratch if necessary.
                """
                
                final_retry_response = await openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": final_retry_prompt}],
                    temperature=1.0,  # Maximum creativity
                    max_tokens=4000
                )
                
                adapted_content = final_retry_response.choices[0].message.content
                adapted_content = adapted_content.replace("```python", "").replace("```", "").strip()
            
            customized_files[file_name] = adapted_content

        # Post-processing validation to ensure consistency between files
        logger.info("Performing cross-file consistency validation...")
        
        # Extract tool class names from agents.py
        if "agents.py" in customized_files and "tools.py" in customized_files:
            tool_imports_in_agents = re.findall(r'from\s+tools\s+import\s+([^#\n]+)', customized_files["agents.py"])
            if tool_imports_in_agents:
                imported_tools = [t.strip() for t in tool_imports_in_agents[0].split(',')]
                logger.info(f"Tool classes imported in agents.py: {imported_tools}")
                
                # Check if these imported tools exist in tools.py
                for tool in imported_tools:
                    if tool.strip() and not re.search(rf'class\s+{re.escape(tool.strip())}\s*\(', customized_files["tools.py"]):
                        logger.warning(f"Tool class '{tool}' imported in agents.py but not defined in tools.py")
                        
                        # Try to correct the imported tool name in agents.py
                        for correct_tool in tool_class_names:
                            if customized_files["agents.py"].replace(tool.strip(), correct_tool):
                                logger.info(f"Corrected tool import in agents.py: {tool} -> {correct_tool}")
                                customized_files["agents.py"] = customized_files["agents.py"].replace(tool.strip(), correct_tool)
                                break
        
        # Check consistency between agents.py and crew.py
        if "agents.py" in customized_files and "crew.py" in customized_files:
            # Extract class names from agents.py
            agent_classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*\(', customized_files["agents.py"])
            
            # Extract imports in crew.py
            agent_imports_in_crew = re.findall(r'from\s+agents\s+import\s+([^#\n]+)', customized_files["crew.py"])
            
            if agent_classes and agent_imports_in_crew:
                imported_agents = [a.strip() for a in agent_imports_in_crew[0].split(',')]
                logger.info(f"Agent classes in agents.py: {agent_classes}")
                logger.info(f"Agent imports in crew.py: {imported_agents}")
                
                # Check for mismatches
                for imported_agent in imported_agents:
                    if imported_agent.strip() and imported_agent.strip() not in agent_classes:
                        logger.warning(f"Agent '{imported_agent}' imported in crew.py but not defined in agents.py")
                        
                        # Try to find the closest matching agent class
                        for agent_class in agent_classes:
                            if 'Factory' in agent_class and 'Factory' not in imported_agent:
                                continue  # Skip factory classes for imported non-factory names
                            
                            # Replace the incorrect import in crew.py
                            if customized_files["crew.py"].replace(imported_agent.strip(), agent_class):
                                logger.info(f"Corrected agent import in crew.py: {imported_agent} -> {agent_class}")
                                customized_files["crew.py"] = customized_files["crew.py"].replace(imported_agent.strip(), agent_class)
                                break
        
        # Check consistency between agents.py and tasks.py
        if "agents.py" in customized_files and "tasks.py" in customized_files:
            # Extract agent factory methods from agents.py
            factory_methods = re.findall(r'def\s+create_([a-zA-Z0-9_]+)', customized_files["agents.py"])
            
            # Extract method calls in tasks.py
            method_calls_in_tasks = re.findall(r'create_([a-zA-Z0-9_]+)', customized_files["tasks.py"])
            
            if factory_methods and method_calls_in_tasks:
                logger.info(f"Factory methods in agents.py: {factory_methods}")
                logger.info(f"Method calls in tasks.py: {method_calls_in_tasks}")
                
                # Check for mismatches
                for method_call in method_calls_in_tasks:
                    if method_call.strip() and method_call.strip() not in factory_methods:
                        logger.warning(f"Method 'create_{method_call}' called in tasks.py but not defined in agents.py")
                        
                        # Try to find the closest matching factory method
                        for factory_method in factory_methods:
                            # Replace the incorrect method call in tasks.py
                            pattern = rf'create_{re.escape(method_call.strip())}'
                            replacement = f'create_{factory_method}'
                            if re.search(pattern, customized_files["tasks.py"]):
                                logger.info(f"Corrected method call in tasks.py: create_{method_call} -> create_{factory_method}")
                                customized_files["tasks.py"] = re.sub(pattern, replacement, customized_files["tasks.py"])
                                break
        
        # Ensure tool class names are consistent across all files
        for file_name, content in customized_files.items():
            for tool_class in tool_class_names:
                # Look for variants of the tool class name in imports
                for file_to_check in customized_files.keys():
                    if file_to_check != "tools.py":
                        # Find tool imports with slight variations
                        variant_pattern = rf'from\s+tools\s+import\s+.*?({re.escape(tool_class[:-4])}[a-zA-Z]*Tool)'
                        variants = re.findall(variant_pattern, customized_files[file_to_check])
                        
                        for variant in variants:
                            if variant != tool_class:
                                logger.warning(f"Found variant tool name import in {file_to_check}: {variant} (should be {tool_class})")
                                customized_files[file_to_check] = customized_files[file_to_check].replace(variant, tool_class)
        
        return customized_files
        
    except Exception as e:
        logger.error(f"Error in direct_requirements_adaptation: {str(e)}")
        return template_files

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Integer distance (lower means more similar)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calculate insertions, deletions and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
