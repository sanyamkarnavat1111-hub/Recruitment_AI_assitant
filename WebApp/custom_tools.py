# email_tool.py
import os
import pickle
import base64
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.tools import tool

SCOPES = ['https://www.googleapis.com/auth/gmail.send']
TOKEN_PATH = "token.pickle"
CREDENTIALS_PATH = "credentials.json"

def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'rb') as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'wb') as f:
            pickle.dump(creds, f)
    return build('gmail', 'v1', credentials=creds)

@tool
def send_email_via_gmail(to: str, subject: str, body: str) -> str:
    """Send email using your Gmail account."""
    try:
        service = get_gmail_service()
        message = {
            'raw': base64.urlsafe_b64encode(
                f"To: {to}\nSubject: {subject}\n\n{body}".encode()
            ).decode()
        }
        result = service.users().messages().send(userId="me", body=message).execute()
        return f"Email sent to {to} (ID: {result['id']})"
    except Exception as e:
        return f"Failed to send email: {str(e)}"