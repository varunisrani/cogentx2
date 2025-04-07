#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run Spotify Agent in different modes')
    parser.add_argument('--mode', type=str, choices=['terminal', 'streamlit'], default='streamlit',
                        help='Mode to run the Spotify Agent (terminal or streamlit)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default='spotify_agent.log', help='Log file path')
    return parser.parse_args()

def run_terminal_mode(args):
    """Run the Spotify Agent in terminal mode"""
    print("Starting Spotify Agent in terminal mode...")
    cmd = [sys.executable, "spotify_agent/main.py"]
    
    if args.verbose:
        cmd.append("--verbose")
    
    if args.log_file:
        cmd.extend(["--log-file", args.log_file])
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nSpotify Agent terminal mode stopped by user.")
    except Exception as e:
        print(f"\nError running Spotify Agent in terminal mode: {e}")
        sys.exit(1)

def run_streamlit_mode(args):
    """Run the Spotify Agent in Streamlit mode"""
    print("Starting Spotify Agent in Streamlit mode...")
    cmd = [sys.executable, "-m", "streamlit", "run", "spotify_streamlit_app.py"]
    
    # Set environment variables for logging
    env = os.environ.copy()
    if args.verbose:
        env["SPOTIFY_VERBOSE"] = "1"
    
    if args.log_file:
        env["SPOTIFY_LOG_FILE"] = args.log_file
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nSpotify Agent Streamlit mode stopped by user.")
    except Exception as e:
        print(f"\nError running Spotify Agent in Streamlit mode: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    
    print("=" * 50)
    print("🎵 SPOTIFY AGENT LAUNCHER 🎵")
    print("=" * 50)
    
    if args.mode == "terminal":
        run_terminal_mode(args)
    else:
        run_streamlit_mode(args)

if __name__ == "__main__":
    main()
