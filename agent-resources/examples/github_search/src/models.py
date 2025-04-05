from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class GitHubRepository(BaseModel):
    """Model for GitHub repository information"""
    name: str = Field(description="Name of the repository")
    full_name: str = Field(description="Full name of the repository (owner/name)")
    description: Optional[str] = Field(description="Repository description")
    url: str = Field(description="Repository URL")
    stars: Optional[int] = Field(description="Number of stars")
    forks: Optional[int] = Field(description="Number of forks")
    language: Optional[str] = Field(description="Primary programming language")
    updated_at: Optional[datetime] = Field(description="Last update timestamp")

    def __str__(self):
        return (
            f"Repository: {self.full_name}\n"
            f"Description: {self.description or 'N/A'}\n"
            f"URL: {self.url}\n"
            f"Stars: {self.stars or 0} | Forks: {self.forks or 0}\n"
            f"Language: {self.language or 'N/A'}\n"
            f"Last Updated: {self.updated_at or 'N/A'}\n"
        )

class GitHubIssue(BaseModel):
    """Model for GitHub issue information"""
    title: str = Field(description="Issue title")
    number: int = Field(description="Issue number")
    state: str = Field(description="Issue state (open/closed)")
    url: str = Field(description="Issue URL")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: Optional[datetime] = Field(description="Last update timestamp")
    body: Optional[str] = Field(description="Issue description")
    labels: List[str] = Field(default_factory=list, description="Issue labels")

    def __str__(self):
        return (
            f"Issue #{self.number}: {self.title}\n"
            f"State: {self.state}\n"
            f"URL: {self.url}\n"
            f"Created: {self.created_at}\n"
            f"Labels: {', '.join(self.labels) if self.labels else 'None'}\n"
            f"Description: {self.body or 'N/A'}\n"
        )

class GitHubSearchResponse(BaseModel):
    """Model for the complete GitHub search response"""
    query: str = Field(description="Original search query")
    repositories: List[GitHubRepository] = Field(default_factory=list, description="List of repository results")
    issues: List[GitHubIssue] = Field(default_factory=list, description="List of issue results")
    summary: str = Field(default="No summary available", description="AI-generated summary of results")

    def __str__(self):
        repos_str = "\n".join(str(r) for r in self.repositories)
        issues_str = "\n".join(str(i) for i in self.issues)
        return (
            f"Query: {self.query}\n\n"
            f"Summary: {self.summary}\n\n"
            f"Repositories:\n{repos_str}\n\n"
            f"Issues:\n{issues_str}"
        )

class GitHubConfig(BaseModel):
    """Configuration for the GitHub agent"""
    access_token: str = Field(description="GitHub Personal Access Token")
    model_name: str = Field(default="gpt-4o-mini", description="Model to use for the agent")
    max_results: int = Field(default=10, description="Maximum number of results to return per search") 