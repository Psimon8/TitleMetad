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
import openai

# Download NLTK resources needed for the script
nltk.download('stopwords')
from nltk.corpus import stopwords

# Define the OAuth 2.0 client secrets file and required scopes
CLIENT_SECRETS_FILE = "client_secrets.json"
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']

def authenticate_user():
    """Authenticate the user using Google OAuth 2.0 and handle token refresh."""
    credentials = load_credentials()
    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            with open('token.pkl', 'wb') as token_file:
                pickle.dump(credentials, token_file)
            st.success("Token refreshed successfully!")
            return credentials
        except Exception as e:
            st.error(f"Error refreshing token: {str(e)}")
            credentials = None

    if not credentials:
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE, scopes=SCOPES,
            redirect_uri="http://localhost:8501"
        )
        authorization_url, _ = flow.authorization_url(prompt='consent')
        st.write("Please authenticate using the following link:")
        st.markdown(f"[Connect with Google]({authorization_url})")
        code = st.text_input("Enter the authorization code here:")

        if code:
            try:
                flow.fetch_token(code=code)
                credentials = flow.credentials
                with open('token.pkl', 'wb') as token_file:
                    pickle.dump(credentials, token_file)
                st.success("Authentication successful!")
                return credentials
            except Exception as e:
                st.error(f"Error during authentication: {str(e)}")
    return credentials

def load_credentials():
    """Load saved credentials if available."""
    if os.path.exists('token.pkl'):
        with open('token.pkl', 'rb') as token_file:
            credentials = pickle.load(token_file)
            if credentials and credentials.valid:
                return credentials
    return None

def get_gsc_service(credentials):
    """Build the Google Search Console (GSC) service using the provided credentials."""
    try:
        service = build('searchconsole', 'v1', credentials=credentials)
        return service
    except HttpError as error:
        st.error(f"Error connecting to the GSC service: {str(error)}")
        return None

def fetch_search_console_data(service, website_url, start_date, end_date, dimensions, dimensionFilterGroups):
    """Fetch GSC data for the specified website and time range."""
    all_responses = []
    start_row = 0

    while True:
        request_body = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": dimensions,
            "dimensionFilterGroups": dimensionFilterGroups,
            "rowLimit": 25000,
            "dataState": "final",
            'startRow': start_row
        }

        try:
            response_data = service.searchanalytics().query(siteUrl=website_url, body=request_body).execute()
            if 'rows' in response_data:
                for row in response_data['rows']:
                    temp = row['keys'] + [row['clicks'], row['impressions'], row['ctr'], row['position']]
                    all_responses.append(temp)

                start_row += len(response_data['rows'])
                if len(response_data['rows']) < 25000:
                    break
            else:
                st.warning("No data available for the selected parameters.")
                break
        except HttpError as error:
            st.error(f"Error fetching data: {str(error)}")
            break

    df = pd.DataFrame(all_responses, columns=dimensions + ['clicks', 'impressions', 'ctr', 'position'])
    return df

def scrape_title_meta_description(url):
    """Scrape and return the title and meta description of a URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.string if soup.title else 'No title found'
    meta_description_tag = soup.find('meta', attrs={'name': 'description'})
    meta_description = meta_description_tag['content'] if meta_description_tag else 'No meta description found'
    return title, meta_description

def identify_gaps(df, selected_url):
    """Identify gaps in the title and meta description based on the fetched GSC data."""
    page_ctr_optimization = df[df['page'] == selected_url]
    page_ctr_optimization_grouped = page_ctr_optimization.groupby('query').agg(
        clicks=('clicks', 'sum'),
        impressions=('impressions', 'sum')
    ).reset_index()
    query_list = page_ctr_optimization_grouped['query'].to_list()

    tokens = []
    for each in query_list:
        each_split = each.split()
        for each_word in each_split:
            if each_word.lower() not in stopwords.words('english'):
                tokens.append(each_word)

    token_counts = Counter(tokens)
    token_counts_df = pd.DataFrame.from_dict(token_counts, orient='index', columns=['count']).reset_index()
    token_counts_df.columns = ['token', 'count']
    token_counts_df = token_counts_df.sort_values(by='count', ascending=False).reset_index(drop=True)

    find_gaps = token_counts_df.head(10)
    return find_gaps['token'].to_list()

def generate_suggestions_for_title_meta(title, meta_description, find_gaps_terms):
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
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt}
            ]
        )
        suggestions = response.choices[0].message['content']
        return suggestions
    except Exception as e:
        st.error(f"Error with OpenAI API: {str(e)}")
        return None

def main():
    """Main function to run the Streamlit app."""
    st.title("Google Search Console Data Analyzer with AI Suggestions")

    # Allow the user to input their OpenAI API key directly in the app
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key

    credentials = authenticate_user()

    if credentials:
        service = get_gsc_service(credentials)
        if service:
            # Use session state to avoid rerun issues
            if 'df' not in st.session_state:
                st.session_state.df = None

            website_url = st.text_input("Enter the website URL (e.g., https://example.com):")
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")

            if st.button("Fetch Data"):
                dimensions = ['date', 'page', 'query']
                dimensionFilterGroups = []
                df = fetch_search_console_data(service, website_url, start_date.strftime("%Y-%m-%d"),
                                               end_date.strftime("%Y-%m-%d"), dimensions, dimensionFilterGroups)
                st.session_state.df = df
                st.write(df)

            # Check if data was fetched successfully before moving on
            if st.session_state.df is not None:
                pattern = st.text_input("Enter URL pattern (e.g., /products/):")
                if pattern:
                    matching_urls = st.session_state.df[st.session_state.df['page'].str.contains(pattern, na=False)]
                    st.write("Matching URLs:", matching_urls)

                    for url in matching_urls['page'].unique():
                        title, meta_description = scrape_title_meta_description(url)
                        st.write(f"URL: {url}")
                        st.write("Title:", title)
                        st.write("Meta Description:", meta_description)

                        find_gaps_terms = identify_gaps(st.session_state.df, url)
                        st.write("Keywords missing in title/meta description:", find_gaps_terms)

                        suggestions = generate_suggestions_for_title_meta(title, meta_description, find_gaps_terms)
                        if suggestions:
                            st.write("AI-generated Suggestions:")
                            st.write(suggestions)

if __name__ == "__main__":
    main()
