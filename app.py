import streamlit as st
import pandas as pd
import os
import pickle
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import nltk
from collections import Counter
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from typing import List, Dict, Any, Optional
import logging
import openai migrate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Constants
CLIENT_SECRETS_FILE = "client_secrets.json"
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
TOKEN_FILE = 'token.pkl'

@st.cache_data
def load_credentials() -> Optional[Credentials]:
    """Load saved credentials if available."""
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'rb') as token_file:
                credentials = pickle.load(token_file)
                if credentials and credentials.valid:
                    return credentials
    except Exception as e:
        logger.error(f"Error loading credentials: {e}")
    return None

def save_credentials(credentials: Credentials) -> None:
    """Save credentials to file."""
    try:
        with open(TOKEN_FILE, 'wb') as token_file:
            pickle.dump(credentials, token_file)
    except Exception as e:
        logger.error(f"Error saving credentials: {e}")

def authenticate_user() -> Optional[Credentials]:
    """Authenticate the user using Google OAuth 2.0 and handle token refresh."""
    credentials = load_credentials()
    
    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            save_credentials(credentials)
            st.success("Token refreshed successfully!")
            return credentials
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            credentials = None

    if not credentials:
        try:
            flow = Flow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, scopes=SCOPES,
                redirect_uri="http://localhost:8501"
            )
            authorization_url, _ = flow.authorization_url(prompt='consent')
            st.write("Please authenticate using the following link:")
            st.markdown(f"[Connect with Google]({authorization_url})")
            code = st.text_input("Enter the authorization code here:")

            if code:
                flow.fetch_token(code=code)
                credentials = flow.credentials
                save_credentials(credentials)
                st.success("Authentication successful!")
                return credentials
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            st.error("Authentication failed. Please try again.")
    
    return credentials

def get_gsc_service(credentials: Credentials):
    """Build the Google Search Console (GSC) service using the provided credentials."""
    try:
        return build('searchconsole', 'v1', credentials=credentials)
    except HttpError as error:
        logger.error(f"Error connecting to the GSC service: {error}")
        st.error(f"Error connecting to the GSC service. Please try again later.")
        return None

@st.cache_data
def fetch_search_console_data(service, website_url: str, start_date: str, end_date: str, 
                              dimensions: List[str], dimension_filter_groups: List[Dict[str, Any]]) -> pd.DataFrame:
    """Fetch GSC data for the specified website and time range."""
    all_responses = []
    start_row = 0

    while True:
        request_body = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": dimensions,
            "dimensionFilterGroups": dimension_filter_groups,
            "rowLimit": 25000,
            "dataState": "final",
            'startRow': start_row
        }

        try:
            response_data = service.searchanalytics().query(siteUrl=website_url, body=request_body).execute()
            if 'rows' in response_data:
                all_responses.extend([row['keys'] + [row['clicks'], row['impressions'], row['ctr'], row['position']] 
                                      for row in response_data['rows']])
                start_row += len(response_data['rows'])
                if len(response_data['rows']) < 25000:
                    break
            else:
                logger.warning("No data available for the selected parameters.")
                break
        except HttpError as error:
            logger.error(f"Error fetching data: {error}")
            st.error("Error fetching data from Google Search Console. Please try again.")
            break

    return pd.DataFrame(all_responses, columns=dimensions + ['clicks', 'impressions', 'ctr', 'position'])

@st.cache_data
def scrape_title_meta_description(url: str) -> tuple:
    """Scrape and return the title and meta description of a URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string if soup.title else 'No title found'
        meta_description_tag = soup.find('meta', attrs={'name': 'description'})
        meta_description = meta_description_tag['content'] if meta_description_tag else 'No meta description found'
        return title, meta_description
    except requests.RequestException as e:
        logger.error(f"Error scraping {url}: {e}")
        return 'Error fetching title', 'Error fetching meta description'

def identify_gaps(df: pd.DataFrame, selected_url: str) -> List[str]:
    """Identify gaps in the title and meta description based on the fetched GSC data."""
    page_data = df[df['page'] == selected_url]
    page_data_grouped = page_data.groupby('query').agg(
        clicks=('clicks', 'sum'),
        impressions=('impressions', 'sum')
    ).reset_index()
    
    tokens = [word.lower() for query in page_data_grouped['query'] 
              for word in query.split() 
              if word.lower() not in stopwords.words('english')]
    
    token_counts = Counter(tokens)
    token_counts_df = pd.DataFrame(token_counts.items(), columns=['token', 'count'])
    token_counts_df = token_counts_df.sort_values(by='count', ascending=False)
    
    return token_counts_df.head(10)['token'].tolist()

def generate_suggestions_for_title_meta(title: str, meta_description: str, find_gaps_terms: List[str]) -> str:
    """Generate optimized suggestions for title and meta description using OpenAI."""
    system_prompt = """
    You are an expert SEO and UX copywriter. Your task is to optimize titles and meta descriptions to increase CTR in SERPs.
    """

    input_prompt = f"""
    Here is the existing Title: {title}
    Here are the gap terms for Title: {find_gaps_terms}

    Here is the existing Meta Description: {meta_description}
    Here are the gap terms for Meta Description: {find_gaps_terms}

    Generate 3 optimized suggestions for both titles and meta descriptions.
    """

    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt}
            ],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error with OpenAI API: {e}")
        st.error("Error generating AI suggestions. Please check your API key and try again.")
        return None

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="GSC Analyzer with GPT-4", page_icon="📊", layout="wide")
    st.title("Google Search Console Data Analyzer with GPT-3.5 Turbo AI Suggestions")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    else:
        st.sidebar.warning("Please enter your OpenAI API key to use AI suggestions.")

    credentials = authenticate_user()

    if credentials:
        service = get_gsc_service(credentials)
        if service:
            website_url = st.text_input("Enter the website URL (e.g., https://example.com):")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")

            if st.button("Fetch Data"):
                with st.spinner("Fetching data from Google Search Console..."):
                    dimensions = ['date', 'page', 'query']
                    dimension_filter_groups = []
                    df = fetch_search_console_data(service, website_url, start_date.strftime("%Y-%m-%d"),
                                                   end_date.strftime("%Y-%m-%d"), dimensions, dimension_filter_groups)
                    st.session_state.df = df
                    st.success("Data fetched successfully!")
                    st.dataframe(df)

            if 'df' in st.session_state and st.session_state.df is not None:
                pattern = st.text_input("Enter URL pattern to analyze (e.g., /products/):")
                if pattern:
                    matching_urls = st.session_state.df[st.session_state.df['page'].str.contains(pattern, na=False)]
                    st.write("Matching URLs:", matching_urls['page'].unique())

                    for url in matching_urls['page'].unique():
                        st.write(f"Analyzing URL: {url}")
                        with st.spinner(f"Scraping and analyzing {url}..."):
                            title, meta_description = scrape_title_meta_description(url)
                            st.write("Current Title:", title)
                            st.write("Current Meta Description:", meta_description)

                            find_gaps_terms = identify_gaps(st.session_state.df, url)
                            st.write("Keywords missing in title/meta description:", find_gaps_terms)

                            suggestions = generate_suggestions_for_title_meta(title, meta_description, find_gaps_terms)
                            if suggestions:
                                st.write("AI-generated Suggestions (GPT-3.5 Turbo):")
                                st.write(suggestions)
        else:
            st.error("Failed to connect to Google Search Console. Please check your credentials and try again.")
    else:
        st.warning("Please authenticate with Google to use this application.")

if __name__ == "__main__":
    main()
