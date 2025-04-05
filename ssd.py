import os
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
import json
from datetime import datetime
from openai import AsyncOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

@dataclass
class AgentTemplate:
    """Dataclass to hold template information"""
    folder_name: str
    agents_code: str
    tools_code: str
    tasks_code: str
    crew_code: str
    purpose: str
    metadata: Dict[str, Any]
    embedding: List[float] = None


class Test18TemplateProcessor:
    """Process the test 18 folder and create embeddings"""
    
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
    
    async def determine_template_purpose(self, agents_code: str, tasks_code: str, crew_code: str, readme_content: str) -> str:
        """Determine the purpose of the agent template using GPT."""
        try:
            prompt = f"""Analyze this market research agent template and provide a concise purpose description.
            
            Agents code excerpt:
            {agents_code[:1000]}...
            
            Tasks code excerpt:
            {tasks_code[:1000]}...
            
            Crew code excerpt:
            {crew_code[:1000]}...
            
            README excerpt:
            {readme_content[:1000]}...
            
            Provide a brief, clear description of what this agent template is designed to do.
            Focus on its main functionality for market research, industry analysis, and report generation.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI expert at analyzing market research agent templates. Provide concise, clear purposes."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error determining purpose: {e}")
            return "Market research template that generates comprehensive reports on products and industries using AI agents with web search capabilities."
    
    def read_file_content(self, file_path: str) -> str:
        """Read the content of a file if it exists"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return file.read()
        return ""
    
    async def process_test18_folder(self) -> AgentTemplate:
        """Process the test 18 folder and return an AgentTemplate object"""
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        
        # Read the required files
        agents_path = os.path.join(base_path, "agents.py")
        tools_path = os.path.join(base_path, "tools.py")
        tasks_path = os.path.join(base_path, "tasks.py")
        crew_path = os.path.join(base_path, "crew.py")
        readme_path = os.path.join(base_path, "README.md")
        
        agents_code = self.read_file_content(agents_path)
        tools_code = self.read_file_content(tools_path)
        tasks_code = self.read_file_content(tasks_path)
        crew_code = self.read_file_content(crew_path)
        readme = self.read_file_content(readme_path)
        
        # Determine purpose using GPT
        purpose = await self.determine_template_purpose(agents_code, tasks_code, crew_code, readme)
        
        # Create metadata
        metadata = {
            "type": "market_research",
            "source": "agent_template",
            "created_at": datetime.utcnow().isoformat(),
            "agents": ["market_analyst", "industry_expert", "report_writer"],
            "default_product": "Smart Home Automation Systems",
            "default_industry": "Consumer Electronics",
            "features": ["comprehensive_market_research", "web_search", "specialized_agents"],
            "description": readme[:500] if readme else "Market research application using CrewAI",
            "has_agents": bool(agents_code),
            "has_tools": bool(tools_code),
            "has_tasks": bool(tasks_code),
            "has_crew": bool(crew_code),
            "has_readme": bool(readme),
            "uses_openai": True,
            "uses_serperdev": True,
            "uses_max_tokens": True
        }
        
        # Create agent template
        template = AgentTemplate(
            folder_name="test 18",
            agents_code=agents_code,
            tools_code=tools_code,
            tasks_code=tasks_code,
            crew_code=crew_code,
            purpose=purpose,
            metadata=metadata,
            embedding=None  # Will be set later
        )
        
        # Create combined text for embedding
        combined_text = f"""
        Purpose: {template.purpose}
        
        Agents:
        {agents_code}
        
        Tools:
        {tools_code}
        
        Tasks:
        {tasks_code}
        
        Crew:
        {crew_code}
        
        Description:
        {readme if readme else ""}
        """
        
        # Get embedding
        template.embedding = await self.get_embedding(combined_text)
        
        return template
    
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
                "tools_code": template.tools_code,
                "tasks_code": template.tasks_code,
                "crew_code": template.crew_code,
                "purpose": template.purpose,
                "metadata": template.metadata,
                "embedding": template.embedding
            }
            
            # Insert into Supabase
            response = self.supabase_client.table("agent_templates").insert(template_dict).execute()
            
            if hasattr(response, 'error') and response.error:
                print(f"Error inserting into database: {response.error}")
            else:
                print("Successfully inserted template into database")
        
        except Exception as e:
            print(f"Error during database insertion: {e}")


async def main():
    """Main function to process the template and insert into database"""
    processor = Test18TemplateProcessor()
    
    print("Processing test 18 folder...")
    template = await processor.process_test18_folder()
    
    print(f"Template purpose determined: {template.purpose}")
    print(f"Generated embedding with {len(template.embedding)} dimensions")
    
    # Save embedding locally as JSON
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template_data.json")
    
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


if __name__ == "__main__":
    asyncio.run(main()) 