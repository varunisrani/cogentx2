from pydantic import BaseModel, Field
from typing import List

class SearchResult(BaseModel):
    """Model for search results"""
    title: str = Field(description="Title of the search result")
    snippet: str = Field(description="Snippet/description of the search result")
    link: str = Field(description="URL of the search result")

    def __str__(self):
        return f"Title: {self.title}\nSnippet: {self.snippet}\nURL: {self.link}\n"

class SearchResponse(BaseModel):
    """Model for the complete search response"""
    query: str = Field(description="Original search query")
    results: List[SearchResult] = Field(description="List of search results")
    summary: str = Field(description="AI-generated summary of results")

    def __str__(self):
        results_str = "\n".join(str(r) for r in self.results)
        return f"Query: {self.query}\nSummary: {self.summary}\nResults:\n{results_str}"

class SerperConfig(BaseModel):
    """Configuration for the Serper agent"""
    serper_api_key: str = Field(description="API key for Serper")
    model_name: str = Field(default="gpt-4o-mini", description="Model to use for the agent")
    temperature: float = Field(default=0.7, description="Temperature for model responses") 