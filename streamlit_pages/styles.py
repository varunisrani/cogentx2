"""
This module contains the CSS styles for the Streamlit UI.
"""

import streamlit as st

def load_css():
    """
    Load the custom CSS styles for the CogentX UI.
    """
    st.markdown("""
        <style>
        :root {
            --primary-color: #1E88E5;  /* Cogentx Blue */
            --secondary-color: #26A69A; /* Cogentx Teal */
            --accent-color: #FF5252;   /* Cogentx Accent Red */
            --background-color: #121212; /* Dark background */
            --card-background: #1E1E1E; /* Dark card background */
            --text-color: #E0E0E0;     /* Light text for dark theme */
            --header-color: #64B5F6;   /* Lighter blue for headers on dark */
            --border-color: #333333;   /* Dark borders */
        }
        
        /* Apply dark background color to the entire app */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        /* Style the buttons */
        .stButton > button {
            color: white !important;
            background-color: var(--primary-color) !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3) !important;
        }
        
        .stButton > button:hover {
            background-color: var(--secondary-color) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.4) !important;
        }
        
        /* Override Streamlit's default focus styles */
        .stButton > button:focus, 
        .stButton > button:focus:hover, 
        .stButton > button:active, 
        .stButton > button:active:hover {
            color: white !important;
            background-color: var(--secondary-color) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.4) !important;
            outline: none !important;
        }
        
        /* Style headers */
        h1 {
            color: var(--header-color) !important;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 1rem !important;
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 0.5rem;
        }
        
        h2 {
            color: var(--header-color) !important;
            font-size: 1.8rem !important;
            font-weight: 600 !important;
        }
        
        h3 {
            color: var(--primary-color) !important;
            font-size: 1.4rem !important;
            font-weight: 600 !important;
        }
        
        /* Hide spans within header elements */
        h1 span, h2 span, h3 span {
            display: none !important;
            visibility: hidden;
            width: 0;
            height: 0;
            opacity: 0;
            position: absolute;
            overflow: hidden;
        }
        
        /* Style code blocks */
        pre {
            border-left: 4px solid var(--primary-color);
            background-color: #2A2A2A !important;
            border-radius: 4px !important;
            color: #E0E0E0 !important;
        }
        
        code {
            color: var(--primary-color) !important;
            background-color: #2A2A2A !important;
        }
        
        /* Style links */
        a {
            color: var(--primary-color) !important;
            text-decoration: none !important;
        }
        
        a:hover {
            color: var(--secondary-color) !important;
            text-decoration: underline !important;
        }
        
        /* Style the chat messages */
        .stChatMessage {
            background-color: var(--card-background) !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
            padding: 12px !important;
            margin-bottom: 16px !important;
            color: var(--text-color) !important;
        }
        
        /* User message styling */
        .stChatMessage[data-testid="stChatMessageUser"] {
            background-color: #1A3A5A !important; /* Dark blue for user */
            border-left: 4px solid var(--primary-color);
        }
        
        /* AI message styling */
        .stChatMessage[data-testid="stChatMessageAssistant"] {
            background-color: var(--card-background) !important;
            border-left: 4px solid var(--secondary-color);
        }
        
        /* Style the chat input */
        .stChatInput > div {
            border: 2px solid var(--primary-color) !important;
            border-radius: 8px !important;
            background-color: var(--card-background) !important;
            color: var(--text-color) !important;
        }
        
        /* Remove outline on focus */
        .stChatInput > div:focus-within {
            box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.3) !important;
            border: 2px solid var(--primary-color) !important;
            outline: none !important;
        }
        
        /* Remove outline on all inputs when focused */
        input:focus, textarea:focus, [contenteditable]:focus {
            box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.3) !important;
            border-color: var(--primary-color) !important;
            outline: none !important;
        }

        /* Style inputs and text areas */
        input, textarea, [contenteditable] {
            background-color: var(--card-background) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
        }
        
        /* Custom styling for tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: var(--background-color) !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            padding: 10px 16px;
            background-color: #333333;
            border: none !important;
            color: var(--text-color) !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary-color) !important;
            color: white !important;
        }

        /* Style selectbox and multiselect */
        .stSelectbox > div, .stMultiSelect > div {
            background-color: var(--card-background) !important;
            color: var(--text-color) !important;
        }
        
        /* Style file upload areas */
        .file-uploader {
            border: 2px dashed var(--primary-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            background-color: rgba(30, 136, 229, 0.1);
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .file-uploader:hover {
            background-color: rgba(30, 136, 229, 0.2);
        }
        
        /* Info boxes */
        .info-box {
            background-color: #1A3A5A;
            border-left: 6px solid var(--primary-color);
            padding: 16px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        
        /* Success boxes */
        .success-box {
            background-color: #1B3B2F;
            border-left: 6px solid var(--secondary-color);
            padding: 16px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        
        /* Warning boxes */
        .warning-box {
            background-color: #3A3215;
            border-left: 6px solid #FFC107;
            padding: 16px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        
        /* Error boxes */
        .error-box {
            background-color: #3B1A1A;
            border-left: 6px solid var(--accent-color);
            padding: 16px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        
        /* Tool section styling */
        .tool-section {
            background-color: var(--card-background);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        
        /* Card styling for containers */
        .stContainer, div[data-testid="stContainer"] {
            background-color: var(--card-background) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
            margin-bottom: 1rem !important;
            border: 1px solid var(--border-color) !important;
        }
        
        /* Style expanders */
        .streamlit-expanderHeader {
            background-color: var(--card-background);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            color: var(--text-color) !important;
        }
        
        .streamlit-expanderContent {
            background-color: var(--card-background);
            border-radius: 0 0 8px 8px;
            border: 1px solid var(--border-color);
            border-top: none;
            color: var(--text-color) !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: var(--card-background);
            border-right: 1px solid var(--border-color);
        }
        
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: var(--header-color) !important;
        }
        
        /* Primary button styling */
        button[data-testid="baseButton-primary"] {
            background-color: var(--primary-color) !important;
            color: white !important;
        }
        
        /* Secondary button styling */
        button[data-testid="baseButton-secondary"] {
            background-color: transparent !important;
            color: var(--primary-color) !important;
            border: 2px solid var(--primary-color) !important;
        }
        
        /* Add Cogentx logo styling */
        .cogentx-logo {
            text-align: center;
            padding: 1rem 0;
        }
        
        .cogentx-logo img {
            max-width: 200px;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-indicator.green {
            background-color: #4CAF50;
        }
        
        .status-indicator.red {
            background-color: #F44336;
        }
        
        .status-indicator.yellow {
            background-color: #FFC107;
        }

        /* Style progress bars */
        div[data-testid="stProgress"] > div {
            background-color: var(--primary-color) !important;
        }

        div[data-testid="stProgress"] {
            background-color: var(--border-color) !important;
        }

        /* Style markdown */
        .element-container div.markdown-text-container {
            color: var(--text-color) !important;
        }

        /* Fix text color in all contexts */
        * {
            color: var(--text-color);
        }
        
        /* Override default Streamlit dark theme elements */
        .st-bq {
            background-color: var(--card-background) !important;
            border-left-color: var(--primary-color) !important;
        }
        
        </style>
    """, unsafe_allow_html=True)
