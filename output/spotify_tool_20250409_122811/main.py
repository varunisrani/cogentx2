#!/usr/bin/env python3
import os
import logging
import datetime
import time
from dotenv import load_dotenv
from crew import SpotifyRecommendationCrew
import argparse

def setup_logging():
    """Set up logging for the Spotify Recommendation Agent system"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/spotify_recommendation_agent_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("spotify_recommendation_agent")
    
    return logger, log_file

def main():
    """Main function to run the Spotify Recommendation Agent system"""
    logger, log_file = setup_logging()
    logger.info("="*80)
    logger.info("Spotify Recommendation Agent System Starting")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    load_dotenv()
    
    spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID')
    spotify_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not spotify_client_id or not spotify_client_secret:
        logger.error("SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET environment variables are missing")
        print("ERROR: Spotify API credentials not found.")
        print("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your .env file.")
        return
    
    logger.info(f"Spotify client ID found: {spotify_client_id[:5]}...{spotify_client_id[-5:] if len(spotify_client_id) > 10 else ''}")
    logger.info(f"Spotify client secret found: {spotify_client_secret[:5]}...{spotify_client_secret[-5:] if len(spotify_client_secret) > 10 else ''}")
    
    parser = argparse.ArgumentParser(description='Spotify Recommendation Agent System')
    parser.add_argument('user_preferences', type=str, nargs='?', help='User preferences for song recommendations')
    parser.add_argument('--no-human-input', action='store_true', help='Disable human input during recommendation execution')
    parser.add_argument('--no-memory', action='store_true', help='Disable memory for the recommendation crew')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    
    args = parser.parse_args()
    logger.info(f"Command-line arguments: {args}")
    
    logger.info("Creating Spotify Recommendation Crew")
    recommendation_crew = SpotifyRecommendationCrew(
        verbose=not args.quiet,
        memory=not args.no_memory,
        human_input=not args.no_human_input
    )
    
    if args.user_preferences:
        logger.info("-"*80)
        logger.info(f"PREFERENCES: {args.user_preferences}")
        print(f"Processing preferences: {args.user_preferences}")
        
        start_time = time.time()
        try:
            recommendations = recommendation_crew.generate_recommendations(args.user_preferences)
            
            execution_time = time.time() - start_time
            logger.info(f"Recommendations completed in {execution_time:.2f} seconds")
            logger.info(f"RESULT LENGTH: {len(recommendations)} characters")
            logger.info(f"RESULT SUMMARY: {recommendations[:100]}..." if len(recommendations) > 100 else recommendations)
            
            print("\nRecommendations:")
            print(recommendations)
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Recommendation process failed after {execution_time:.2f} seconds: {str(e)}")
            print(f"Error generating recommendations: {str(e)}")
    else:
        logger.info("Starting interactive mode")
        print("Spotify Recommendation Agent System")
        print("-------------------------------")
        print("Enter 'exit' or 'quit' to exit.")
        print("Enter your song preferences below:")
        
        while True:
            try:
                preferences = input("\nPreferences: ").strip()
                if preferences.lower() in ['exit', 'quit']:
                    logger.info("User requested exit")
                    break
                
                if preferences:
                    logger.info("-"*80)
                    logger.info(f"INTERACTIVE PREFERENCES: {preferences}")
                    
                    start_time = time.time()
                    recommendations = recommendation_crew.generate_recommendations(preferences)
                    
                    execution_time = time.time() - start_time
                    logger.info(f"Recommendations completed in {execution_time:.2f} seconds")
                    logger.info(f"RESULT LENGTH: {len(recommendations)} characters")
                    logger.info(f"RESULT SUMMARY: {recommendations[:100]}..." if len(recommendations) > 100 else recommendations)
                    
                    print("\nRecommendations:")
                    print(recommendations)
            except KeyboardInterrupt:
                logger.info("Interrupted by user (CTRL+C)")
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error processing interactive preferences: {str(e)}")
                print(f"Error: {str(e)}")
    
    logger.info("="*80)
    logger.info("Spotify Recommendation Agent System Shutting Down")
    logger.info("="*80)

if __name__ == "__main__":
    main()