import streamlit as st
import pandas as pd
import os
import pickle
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import nltk
from collections import Counter
import requests
from bs4 import BeautifulSoup

nltk.download('stopwords')
from nltk.corpus import stopwords

# Define OAuth 2.0 Client Secrets file
CLIENT_SECRETS_FILE = "client_secrets.json"
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']

def authenticate_user():
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
    return None

def load_credentials():
    if os.path.exists('token.pkl'):
        with open('token.pkl', 'rb') as token_file:
            credentials = pickle.load(token_file)
            return credentials
    return None

def get_gsc_service(credentials):
    try:
        service = build('searchconsole', 'v1', credentials=credentials)
        return service
    except HttpError as error:
        st.error(f"Error connecting to the GSC service: {str(error)}")
        return None

def get_site_list(service):
    try:
        site_list = service.sites().list().execute()
        return [site['siteUrl'] for site in site_list.get('siteEntry', [])]
    except HttpError as error:
        st.error(f"Error fetching site list: {str(error)}")
        return []

def fetch_search_console_data(service, website_url, start_date, end_date, dimensions, dimensionFilterGroups):
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
            for row in response_data['rows']:
                temp = row['keys'] + [row['clicks'], row['impressions'], row['ctr'], row['position']]
                all_responses.append(temp)

            start_row += len(response_data['rows'])
            if len(response_data['rows']) < 25000:
                break
        except HttpError as error:
            st.error(f"Error fetching data: {str(error)}")
            break

    df = pd.DataFrame(all_responses, columns=dimensions + ['clicks', 'impressions', 'ctr', 'position'])
    return df

def scrape_title_meta_description(url):
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

def main():
    st.title("Google Search Console Data Analyzer")

    credentials = load_credentials()
    if not credentials:
        credentials = authenticate_user()

    if credentials:
        service = get_gsc_service(credentials)
        if service:
            sites = get_site_list(service)
            if sites:
                selected_site = st.selectbox("Select a site to analyze:", sites)
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date")

                if st.button("Fetch Data"):
                    dimensions = ['date', 'page', 'query']
                    dimensionFilterGroups = []
                    df = fetch_search_console_data(service, selected_site, start_date.strftime("%Y-%m-%d"),
                                                   end_date.strftime("%Y-%m-%d"), dimensions, dimensionFilterGroups)
                    st.write(df)

                    selected_url = st.text_input("Enter URL to analyze:")
                    if selected_url:
                        title, meta_description = scrape_title_meta_description(selected_url)
                        st.write("Title:", title)
                        st.write("Meta Description:", meta_description)

                        find_gaps_terms = identify_gaps(df, selected_url)
                        st.write("Keywords missing in title/meta description:", find_gaps_terms)

if __name__ == "__main__":
    main()
