#!/usr/bin/env python3
import os
import asyncio
from supabase import create_client
from dotenv import load_dotenv

def setup_supabase_database():
    """Set up the Supabase database with the required tables and functions"""
    # Load environment variables
    load_dotenv()
    
    # Get Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in your .env file")
        return False
    
    try:
        # Initialize Supabase client
        print("Connecting to Supabase...")
        supabase = create_client(supabase_url, supabase_key)
        
        # Read the SQL file
        print("Reading SQL setup file...")
        with open("create_table.sql", "r") as f:
            sql_script = f.read()
        
        # Split the SQL script by semicolons to get individual statements
        sql_statements = [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]
        
        # Execute each SQL statement
        for i, statement in enumerate(sql_statements, 1):
            if not statement:
                continue
                
            print(f"Executing SQL statement {i} of {len(sql_statements)}...")
            try:
                # First, ensure pgvector extension is installed
                if i == 1:  # Only do this for the first statement
                    print("Ensuring pgvector extension is enabled...")
                    try:
                        supabase.rpc(
                            "pg_query",
                            {
                                "query": "CREATE EXTENSION IF NOT EXISTS vector;"
                            }
                        ).execute()
                        print("pgvector extension enabled successfully.")
                    except Exception as e:
                        print(f"Warning: Could not enable pgvector extension: {e}")
                        print("You may need to enable it manually in the Supabase dashboard.")
                
                # Execute the SQL statement using Supabase SQL API
                response = supabase.rpc(
                    "pg_query",
                    {
                        "query": statement
                    }
                ).execute()
                
                print(f"Statement {i} executed successfully.")
            except Exception as e:
                print(f"Error executing statement {i}: {e}")
                print("Statement:", statement)
                # Continue with other statements
        
        print("\nDatabase setup completed. You should now be able to search for agent templates.")
        return True
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        return False

def main():
    """Main function"""
    print("Setting up Supabase database for agent template search...")
    result = setup_supabase_database()
    
    if result:
        print("\nDatabase setup complete. Try searching for templates:")
        print("python search_templates.py \"agent\" --threshold 0.3")
    else:
        print("\nDatabase setup failed. Please check your Supabase credentials and permissions.")
        print("Make sure you have enabled the pgvector extension in your Supabase project.")

if __name__ == "__main__":
    main() 