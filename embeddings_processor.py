import os
import asyncio
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from openai import AsyncOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

@dataclass
class AgentTemplate:
    """Dataclass to hold agent template information"""
    folder_name: str
    agents_code: str
    main_code: str
    models_code: str
    tools_code: str
    mcp_json: str
    purpose: str
    metadata: Dict[str, Any]
    embedding: List[float] = None


class AgentEmbeddingProcessor:
    """Process agent files, create embeddings, and store in database"""
    
    def __init__(self):
        """Initialize the OpenAI and Supabase clients"""
        load_dotenv()
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize Supabase client
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        if self.supabase_url and self.supabase_key:
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
        else:
            print("Warning: Supabase credentials not found. Database operations will be skipped.")
            self.supabase_client = None
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from OpenAI"""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0] * 1536
    
    async def determine_template_purpose(self, agents_code: str, main_code: str, models_code: str, 
                                        tools_code: str, mcp_json: str, readme: str = "") -> str:
        """Determine the purpose of the agent template using GPT."""
        try:
            prompt = f"""Analyze this agent template and provide a concise purpose description.
            
            Agent code excerpt:
            {agents_code[:800]}...
            
            Main code excerpt:
            {main_code[:800]}...
            
            Models code excerpt:
            {models_code[:800]}...
            
            Tools code excerpt:
            {tools_code[:800]}...
            
            README excerpt:
            {readme[:800]}...
            
            Provide a brief, clear description of what this agent template is designed to do.
            Focus on its main functionality, purpose, and capabilities.
            Keep it under 100 words.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI expert at analyzing agent templates. Provide concise, clear purposes."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error determining purpose: {e}")
            return "Agent template for automating tasks with AI capabilities."
    
    def read_file_content(self, file_path: str) -> str:
        """Read the content of a file if it exists"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return file.read()
        return ""
    
    async def process_agent_folder(self, folder_path: str) -> AgentTemplate:
        """Process an agent folder and return an AgentTemplate object"""
        # Read the required files
        agents_path = os.path.join(folder_path, "agent.py")
        main_path = os.path.join(folder_path, "main.py")
        models_path = os.path.join(folder_path, "models.py")
        tools_path = os.path.join(folder_path, "tools.py")
        mcp_path = os.path.join(folder_path, "mcp.json")
        readme_path = os.path.join(folder_path, "README.md")
        requirements_path = os.path.join(folder_path, "requirements.txt")
        
        agents_code = self.read_file_content(agents_path)
        main_code = self.read_file_content(main_path)
        models_code = self.read_file_content(models_path)
        tools_code = self.read_file_content(tools_path)
        mcp_json = self.read_file_content(mcp_path)
        readme = self.read_file_content(readme_path)
        requirements = self.read_file_content(requirements_path)
        
        # Determine folder name from path
        folder_name = os.path.basename(os.path.normpath(folder_path))
        
        # Determine purpose using GPT
        purpose = await self.determine_template_purpose(
            agents_code, main_code, models_code, tools_code, mcp_json, readme
        )
        
        # Extract agent names, capabilities, and dependencies
        agent_names = []
        agent_descriptions = []
        capabilities = []
        dependencies = []
        
        # Extract dependencies from requirements.txt
        if requirements:
            for line in requirements.strip().split('\n'):
                if line and not line.startswith('#'):
                    dependencies.append(line.split('==')[0] if '==' in line else line)
        
        # Extract agent info using regex
        import re
        
        # Extract agent classes
        if agents_code:
            # Extract class definitions
            class_matches = re.findall(r'class\s+(\w+)', agents_code)
            agent_names.extend(class_matches)
            
            # Try to extract agent descriptions from docstrings
            for agent_name in agent_names:
                class_pattern = rf'class\s+{agent_name}[^\n]*:(.*?)(?:def|\Z)'
                class_match = re.search(class_pattern, agents_code, re.DOTALL)
                if class_match:
                    class_content = class_match.group(1)
                    docstring_match = re.search(r'"""(.*?)"""', class_content, re.DOTALL)
                    if docstring_match:
                        description = docstring_match.group(1).strip()
                        agent_descriptions.append({
                            "name": agent_name,
                            "description": ' '.join(description.split())[:200]  # Truncate long descriptions
                        })
        
        # Extract tool functions
        if tools_code:
            func_matches = re.findall(r'def\s+(\w+)', tools_code)
            capabilities.extend(func_matches)
            
            # Extract tool imports - potential external dependencies
            import_matches = re.findall(r'import\s+(\w+)', tools_code)
            from_import_matches = re.findall(r'from\s+(\w+)', tools_code)
            
            potential_deps = set(import_matches + from_import_matches)
            potential_deps = {dep for dep in potential_deps if dep not in ['os', 'sys', 'json', 're', 'typing', 'abc', 'dataclasses', 'enum']}
            
            if potential_deps:
                dependencies.extend(list(potential_deps))
        
        # Create metadata
        metadata = {
            "agent_type": folder_name.replace("_agent", ""),
            "source": "agent_template",
            "created_at": datetime.utcnow().isoformat(),
            "agents": agent_names,
            "agent_descriptions": agent_descriptions if agent_descriptions else None,
            "capabilities": capabilities,
            "dependencies": list(set(dependencies)),  # Deduplicate
            "features": self.extract_features(agents_code, tools_code, models_code, main_code),
            "has_agents": bool(agents_code),
            "has_main": bool(main_code),
            "has_models": bool(models_code),
            "has_tools": bool(tools_code),
            "has_mcp": bool(mcp_json),
            "has_readme": bool(readme),
            "file_sizes": {
                "agent.py": len(agents_code),
                "main.py": len(main_code),
                "models.py": len(models_code),
                "tools.py": len(tools_code),
                "mcp.json": len(mcp_json) if mcp_json else 0,
                "readme.md": len(readme) if readme else 0,
                "requirements.txt": len(requirements) if requirements else 0
            }
        }
        
        # Try to parse MCP JSON for additional metadata
        if mcp_json:
            try:
                mcp_data = json.loads(mcp_json)
                if isinstance(mcp_data, dict):
                    metadata["mcp_keys"] = list(mcp_data.keys())
                    metadata["mcp_data"] = mcp_data
            except json.JSONDecodeError:
                pass
        
        # Create agent template
        template = AgentTemplate(
            folder_name=folder_name,
            agents_code=agents_code,
            main_code=main_code,
            models_code=models_code,
            tools_code=tools_code,
            mcp_json=mcp_json,
            purpose=purpose,
            metadata=metadata,
            embedding=None  # Will be set later
        )
        
        # Create combined text for embedding
        combined_text = f"""
        Purpose: {template.purpose}
        
        Agents: {', '.join(agent_names)}
        
        Agent Code:
        {agents_code[:2000]}
        
        Main Code:
        {main_code[:2000]}
        
        Models Code:
        {models_code[:2000]}
        
        Tools Code:
        {tools_code[:2000]}
        
        Description:
        {readme[:500] if readme else ""}
        
        Capabilities: {', '.join(capabilities[:20])}
        """
        
        # Get embedding
        template.embedding = await self.get_embedding(combined_text)
        
        return template
    
    def extract_features(self, agents_code: str, tools_code: str, models_code: str, main_code: str) -> List[str]:
        """Extract features from code files"""
        features = []
        
        # Check for common features
        if 'OpenAI' in tools_code or 'openai' in tools_code:
            features.append("uses_openai")
        
        if 'anthropic' in tools_code:
            features.append("uses_anthropic")
            
        if 'serper' in tools_code or 'serper' in main_code:
            features.append("uses_serper")
            
        if 'google.generativeai' in tools_code:
            features.append("uses_gemini")
            
        if 'github' in tools_code:
            features.append("github_integration")
            
        if 'flask' in main_code:
            features.append("web_api")
            
        if 'spotify' in tools_code:
            features.append("spotify_integration")
            
        if 'firecrawl' in tools_code or 'selenium' in tools_code:
            features.append("web_crawling")
            
        if 'asyncio' in tools_code or 'async def' in tools_code:
            features.append("async_capabilities")
            
        return features
    
    async def insert_template(self, template: AgentTemplate) -> None:
        """Insert the template into Supabase database"""
        if not self.supabase_client:
            print("Skipping database insertion: Supabase client not initialized")
            return
        
        try:
            # Convert template to dict for database insertion
            template_dict = {
                "folder_name": template.folder_name,
                "agents_code": template.agents_code,
                "main_code": template.main_code,
                "models_code": template.models_code,
                "tools_code": template.tools_code,
                "mcp_json": template.mcp_json,
                "purpose": template.purpose,
                "metadata": template.metadata,
                "embedding": template.embedding
            }
            
            # Insert into Supabase
            response = self.supabase_client.table("agent_embeddings").insert(template_dict).execute()
            
            if hasattr(response, 'error') and response.error:
                print(f"Error inserting into database: {response.error}")
            else:
                print("Successfully inserted template into database")
        
        except Exception as e:
            print(f"Error during database insertion: {e}")
    
    async def search_similar_templates(self, query_text: str, similarity_threshold: float = 0.5, match_count: int = 5) -> List[Dict]:
        """Search for similar templates using embedding similarity"""
        if not self.supabase_client:
            print("Cannot search: Supabase client not initialized")
            return []
        
        try:
            # Get embedding for the query text
            query_embedding = await self.get_embedding(query_text)
            
            # Search using the search_agent_embeddings function
            response = self.supabase_client.rpc(
                'search_agent_embeddings', 
                {
                    'query_embedding': query_embedding,
                    'similarity_threshold': similarity_threshold,
                    'match_count': match_count
                }
            ).execute()
            
            if hasattr(response, 'error') and response.error:
                print(f"Error searching templates: {response.error}")
                return []
            
            return response.data
            
        except Exception as e:
            print(f"Error searching templates: {e}")
            return []
            
    async def save_template_metadata(self, template: AgentTemplate, output_path: str) -> None:
        """Save template metadata to a JSON file"""
        try:
            # Convert dataclass to dict for JSON serialization
            template_dict = {
                "folder_name": template.folder_name,
                "purpose": template.purpose,
                "metadata": template.metadata,
                "embedding_size": len(template.embedding) if template.embedding else 0,
                "embedding_sample": template.embedding[:10] + ["..."] if template.embedding else [],  # Truncate for readability
                "timestamp": datetime.utcnow().isoformat(),
                "file_sizes": {
                    "agent.py": len(template.agents_code) if template.agents_code else 0,
                    "main.py": len(template.main_code) if template.main_code else 0,
                    "models.py": len(template.models_code) if template.models_code else 0,
                    "tools.py": len(template.tools_code) if template.tools_code else 0,
                    "mcp.json": len(template.mcp_json) if template.mcp_json else 0
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(template_dict, f, indent=2)
                
            print(f"Template metadata saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving template metadata: {e}")


async def main():
    """Main function to process agent templates and insert into database"""
    processor = AgentEmbeddingProcessor()
    
    # Get folder path from command line or use default
    import sys
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Processing agent folder: {folder_path}")
    template = await processor.process_agent_folder(folder_path)
    
    print(f"Template purpose determined: {template.purpose}")
    print(f"Generated embedding with {len(template.embedding)} dimensions")
    
    # Save embedding locally as JSON
    output_path = os.path.join(folder_path, "template_data.json")
    
    # Convert dataclass to dict for JSON serialization
    template_dict = {
        "folder_name": template.folder_name,
        "purpose": template.purpose,
        "metadata": template.metadata,
        "embedding_size": len(template.embedding),
        "embedding_sample": template.embedding[:10] + ["..."] if template.embedding else []  # Truncate for readability
    }
    
    with open(output_path, 'w') as f:
        json.dump(template_dict, f, indent=2)
    
    print(f"Template data saved to {output_path}")
    
    # Insert into database if credentials are available
    if processor.supabase_client:
        print("Inserting template into database...")
        await processor.insert_template(template)
        print("Completed processing and database insertion")
    else:
        print("Skipping database insertion due to missing credentials")
        print("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file to enable database storage.")
    
    # Example of searching for similar templates
    if processor.supabase_client:
        print("\nSearching for similar templates...")
        search_query = f"Purpose: {template.purpose}"
        similar_templates = await processor.search_similar_templates(search_query)
        
        print(f"Found {len(similar_templates)} similar templates:")
        for idx, t in enumerate(similar_templates):
            print(f"{idx+1}. {t['folder_name']} (similarity: {t['similarity']:.4f})")
            print(f"   Purpose: {t['purpose']}")
            print()


if __name__ == "__main__":
    asyncio.run(main()) 