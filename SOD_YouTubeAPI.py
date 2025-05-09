"""
YouTube API Integration Module
Handles YouTube API for broadcast management, enabling stream rotation.
"""

import os
import time
import datetime
import json
from typing import Optional, Dict, Any, Tuple

# Try to import the Google API client
try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    print("Google API client not installed. Install with:")
    print("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

# YouTube API scopes
SCOPES = ['https://www.googleapis.com/auth/youtube']

# Persistent stream ID file
STREAM_ID_FILE = 'youtube_stream_id.json'

class YouTubeAPIManager:
    """Manages YouTube API access and broadcast operations."""
    
    def __init__(self):
        """Initialize the YouTube API manager."""
        self.api = None
        self.authorized = False
        self.stream_id = None
        self.current_broadcast_id = None
        
        # Try to load credentials and connect to API
        if GOOGLE_API_AVAILABLE:
            self._load_credentials()
            
    def _load_credentials(self) -> bool:
        """
        Load or refresh YouTube API credentials.
        
        Returns:
            bool: True if successful, False otherwise
        """
        creds = None
        token_path = 'youtube_token.json'
        client_secrets_path = 'client_secrets.json'
        
        # Check if we have saved token
        if os.path.exists(token_path):
            try:
                creds = Credentials.from_authorized_user_info(
                    json.loads(open(token_path, 'r').read()), SCOPES)
            except Exception as e:
                print(f"Error loading credentials: {str(e)}")
                
        # If credentials are invalid or don't exist, refresh or create new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except RefreshError:
                    print("Token refresh failed, need to re-authenticate")
                    creds = None
            
            # If still no valid credentials, need to authenticate        
            if not creds:
                if os.path.exists(client_secrets_path):
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            client_secrets_path, SCOPES)
                        creds = flow.run_local_server(port=0)
                        
                        # Save the credentials for the next run
                        with open(token_path, 'w') as token:
                            token.write(creds.to_json())
                    except Exception as e:
                        print(f"Error during authorization: {str(e)}")
                        return False
                else:
                    print(f"Client secrets file '{client_secrets_path}' not found.")
                    print("Download this file from the Google Cloud Console:")
                    print("https://console.cloud.google.com/apis/credentials")
                    return False
        
        # Create YouTube API client
        try:
            self.api = build('youtube', 'v3', credentials=creds)
            self.authorized = True
            return True
        except Exception as e:
            print(f"Failed to build YouTube API client: {str(e)}")
            return False
    
    def load_or_create_stream(self, stream_title: str = "ISS UAP Detector Stream") -> bool:
        """
        Load existing stream ID or create a new reusable stream.
        
        Args:
            stream_title: Title for the stream
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.authorized or not self.api:
            print("YouTube API not authorized")
            return False
            
        # Try to load existing stream ID from file
        if os.path.exists(STREAM_ID_FILE):
            try:
                with open(STREAM_ID_FILE, 'r') as f:
                    data = json.load(f)
                    self.stream_id = data.get('stream_id')
                    print(f"Loaded existing stream ID: {self.stream_id}")
                    return True
            except Exception as e:
                print(f"Error loading stream ID: {str(e)}")
        
        # Create a new stream if needed
        if not self.stream_id:
            try:
                # Create a new livestream
                stream_request = self.api.liveStreams().insert(
                    part="snippet,cdn",
                    body={
                        "snippet": {
                            "title": stream_title
                        },
                        "cdn": {
                            "frameRate": "60fps",
                            "ingestionType": "rtmp",
                            "resolution": "1080p"
                        }
                    }
                )
                stream_response = stream_request.execute()
                self.stream_id = stream_response['id']
                
                # Save the stream ID for future use
                with open(STREAM_ID_FILE, 'w') as f:
                    json.dump({'stream_id': self.stream_id}, f)
                    
                print(f"Created new YouTube stream with ID: {self.stream_id}")
                return True
                
            except Exception as e:
                print(f"Error creating YouTube stream: {str(e)}")
                return False
                
        return False
        
    def create_broadcast(self, title: Optional[str] = None, privacy: str = "public") -> Optional[str]:
        """
        Create a new broadcast and bind it to the reusable stream.
        
        Args:
            title: Optional title for the broadcast. If None, auto-generates one with timestamp.
            privacy: Privacy status ('public', 'private', or 'unlisted')
            
        Returns:
            str: Broadcast ID if successful, None otherwise
        """
        if not self.authorized or not self.api:
            print("YouTube API not authorized")
            return None
            
        if not self.stream_id:
            print("No stream ID available. Call load_or_create_stream() first.")
            return None
            
        # Generate title with timestamp if not provided
        if not title:
            now = datetime.datetime.now()
            title = f"ISS UAP Detector Stream {now.strftime('%Y-%m-%d %H:%M')}"
            
        try:
            # Create a new broadcast
            now = datetime.datetime.utcnow()
            broadcast_request = self.api.liveBroadcasts().insert(
                part="snippet,status,contentDetails",
                body={
                    "snippet": {
                        "title": title,
                        "scheduledStartTime": (now + datetime.timedelta(minutes=1)).isoformat("T") + "Z"
                    },
                    "status": {
                        "privacyStatus": privacy
                    },
                    "contentDetails": {
                        "enableAutoStart": True,
                        "enableAutoStop": True,
                        "recordFromStart": True,
                        "enableDvr": True
                    }
                }
            )
            broadcast_response = broadcast_request.execute()
            broadcast_id = broadcast_response['id']
            
            # Bind the broadcast to our reusable stream
            bind_request = self.api.liveBroadcasts().bind(
                part="id",
                id=broadcast_id,
                streamId=self.stream_id
            )
            bind_request.execute()
            
            self.current_broadcast_id = broadcast_id
            print(f"Created and bound new broadcast: {title} (ID: {broadcast_id})")
            return broadcast_id
            
        except Exception as e:
            print(f"Error creating broadcast: {str(e)}")
            return None
            
    def get_stream_key(self) -> Optional[str]:
        """
        Get the stream key for the current stream.
        
        Returns:
            str: Stream key if available, None otherwise
        """
        if not self.authorized or not self.api or not self.stream_id:
            return None
            
        try:
            request = self.api.liveStreams().list(
                part="cdn",
                id=self.stream_id
            )
            response = request.execute()
            
            if 'items' in response and len(response['items']) > 0:
                stream_key = response['items'][0]['cdn']['ingestionInfo']['streamName']
                return stream_key
            else:
                print("Stream details not found")
                return None
                
        except Exception as e:
            print(f"Error retrieving stream key: {str(e)}")
            return None

    def rotate_broadcast(self) -> Tuple[bool, Optional[str]]:
        """
        Create a new broadcast for stream rotation.
        
        Returns:
            Tuple[bool, Optional[str]]: (Success flag, New broadcast ID if successful)
        """
        new_broadcast_id = self.create_broadcast()
        return (new_broadcast_id is not None, new_broadcast_id) 