from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
from builtins import print
import os
import re
import json
import urllib.parse
from webexteamssdk import WebexTeamsAPI
from datetime import datetime, timedelta, date
import time
import dateutil.parser
from requests.adapters import HTTPAdapter, Retry


class WebexMiner:
    proxy = {
        "http": "http://internet.ford.com:83",
        "https": "http://internet.ford.com:83"
    }

    def __init__(self, token):
        """
        Initializes the WebexMiner with the provided access token.

        Args:
            token (str): The access token for authenticating with the Webex Teams API.
        Returns:
            str: The extracted filename, or None if no filename is found.
        """
        if not token:
            raise ValueError("Token must not be None")
        self.api = WebexTeamsAPI(access_token=token, proxies=self.proxy)
        
        self.current_person_id = self.api.people.me().id
    
    def _sanitize_filename(self, filename):
        """
        Sanitizes a string to be used as a filename.
        """
        return ''.join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
    
    def _get_filename_from_file_url(self, file_url):
        """
        Extracts filename from the file URL or sets a default filename.
        """
        parsed_url = urllib.parse.urlparse(file_url)
        filename = os.path.basename(parsed_url.path)
        if filename:
            return filename
        else:
            return 'attachment'
    
    @staticmethod
    def _get_filename_from_cd(content_disposition):
        """
        Extracts filename from Content-Disposition header.
        """
        if not content_disposition:
            return None
        fname = re.findall(r'filename\*=UTF-8\'\'(.+)', content_disposition)
        if fname:
            fname = urllib.parse.unquote(fname[0])
            return fname
        fname = re.findall(r'filename="?([^\";]+)"?', content_disposition)
        if fname:
            return fname[0]
        return None

    def _escape_html(self, text):
        """
        Escapes HTML special characters in a text string.
        """
        import html
        return html.escape(text or '')

    def _generate_html_chat(self, messages_data, html_file):
        """
        Generates an HTML file displaying the chat history in a messaging app style,
        with in-page previews of attachments when possible.
        """
        # Basic HTML template with inline CSS for styling
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Chat History</title>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f5f5f5; }}
                .chat-container {{ max-width: 800px; margin: auto; padding: 20px; background-color: #fff; }}
                .message {{ margin-bottom: 15px; }}
                .message .sender {{ font-weight: bold; }}
                .message .timestamp {{ color: #999; font-size: 0.9em; }}
                .message .text {{ margin: 5px 0; }}
                .message.own {{ text-align: right; }}
                .message.own .message-content {{ display: inline-block; text-align: left; background-color: #dcf8c6; padding: 10px; border-radius: 10px; }}
                .message.other .message-content {{ display: inline-block; background-color: #e0e0e0; padding: 10px; border-radius: 10px; }}
                .attachment {{ margin-top: 5px; }}
                .attachment img, .attachment video, .attachment audio {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="chat-container">
                {messages}
            </div>
        </body>
        </html>
        """

        # Build the messages HTML
        messages_html = ''
        for msg in messages_data:
            # Determine if the message is from the current user or others
            own_message = (msg['personId'] == self.current_person_id)
            sender_class = 'own' if own_message else 'other'

            # Format timestamp
            timestamp = msg['created'].strftime('%Y-%m-%d %H:%M:%S')

            # Escape HTML special characters in text
            message_text = self._escape_html(msg['text'])

            # Build attachments HTML
            attachments_html = ''
            for file_name in msg['files']:
                # Determine the file type based on the file extension
                file_ext = os.path.splitext(file_name)[1].lower()
                preview_html = ''
                if file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                    # Image preview
                    preview_html = f'<img src="{file_name}" alt="Image">'
                elif file_ext in ['.mp4', '.webm', '.ogg']:
                    # Video preview
                    preview_html = f'''
                    <video controls>
                        <source src="{file_name}" type="video/{file_ext[1:]}">
                        Your browser does not support the video tag.
                    </video>
                    '''
                elif file_ext in ['.mp3', '.wav', '.ogg']:
                    # Audio preview
                    preview_html = f'''
                    <audio controls>
                        <source src="{file_name}" type="audio/{file_ext[1:]}">
                        Your browser does not support the audio element.
                    </audio>
                    '''
                elif file_ext == '.pdf':
                    # PDF preview
                    preview_html = f'''
                    <embed src="{file_name}" type="application/pdf" width="100%" height="600px" />
                    '''
                else:
                    # For other file types, provide a download link
                    preview_html = f'<a href="{file_name}" download>{file_name}</a>'
                attachments_html += f'<div class="attachment">{preview_html}</div>'

            # Build message HTML
            message_html = f"""
            <div class="message {sender_class}">
                <div class="message-content">
                    <div class="sender">{self._escape_html(msg['displayName'])}</div>
                    <div class="timestamp">{timestamp}</div>
                    <div class="text">{message_text}</div>
                    {attachments_html}
                </div>
            </div>
            """

            messages_html += message_html

        # Generate final HTML by inserting messages into the template
        final_html = html_template.format(messages=messages_html)

        # Write the HTML file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(final_html)
    
    def _export_room_messages(self, room, output_dir):
        """
        Exports messages and attachments from a Webex room, handling pagination to retrieve all messages.
        Generates an HTML file formatted like a messaging app for easy readability.
        """
        # Sanitize room title for directory name
        room_title = room.title if hasattr(room, 'title') else 'Direct Message'
        sanitized_title = self._sanitize_filename(room_title)
        room_dir = os.path.join(output_dir, f"{sanitized_title}_{room.id}")
        if not os.path.exists(room_dir):
            os.makedirs(room_dir)
        
        # File to store messages
        messages_file = os.path.join(room_dir, 'messages.json')
        html_file = os.path.join(room_dir, 'chat_history.html')
        messages_data = []
        
        print(f"Exporting messages from room: {room_title}")

        # Retrieve messages, handling pagination
        max_items = 200  # Max items per page as per Webex API
        last_message_id = None
        user_cache = {}

        while True:
            # Set up parameters for the API call
            if last_message_id:
                # Fetch messages before the last message ID obtained
                messages = self.api.messages.list(
                    roomId=room.id,
                    beforeMessage=last_message_id,
                    max=max_items
                )
            else:
                # Fetch the first page of messages
                messages = self.api.messages.list(
                    roomId=room.id,
                    max=max_items
                )

            retry_attempts = 3
            failed = False
            for attempt in range(retry_attempts):
                try:
                    messages_page = list(messages)
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < retry_attempts - 1:
                        time.sleep(5 ** attempt)  # Exponential backoff
                    else: 
                        print(f"Access denied when retrieving messages for room: {room.title} ({room.id}). Skipping this room.")
                        failed = True
                        break
            if failed:
                break
            if not messages_page:
                # No more messages to retrieve
                break

            # Process messages in current page
            for message in reversed(messages_page):  # Reverse to get messages in chronological order
                message_data = {
                    'id': message.id,
                    'roomId': message.roomId,
                    'roomType': message.roomType,
                    'text': message.text or '',
                    'personId': message.personId,
                    'personEmail': message.personEmail,
                    'created': message.created,
                    'files': [],
                    'displayName': ''
                }
                
                # Get sender's display name (cached to minimize API calls)
                if message.personId in user_cache:
                    message_data['displayName'] = user_cache[message.personId]
                else:
                    try:
                        person = self.api.people.get(message.personId)
                        display_name = person.displayName
                    except Exception:
                        display_name = message.personEmail  # Fallback to email if name not available
                    user_cache[message.personId] = display_name
                    message_data['displayName'] = display_name

                # Handle attachments
                if hasattr(message, 'files') and message.files:
                    # Download each attachment
                    for file_url in message.files:
                        print(f"Downloading file from URL: {file_url}")
                        try:
                            # Get the filename from Content-Disposition
                            # Use the 'request' method instead of 'get' to avoid JSON parsing
                            response = self.api._session.request('GET', file_url, 200, stream=True)
                            filename = None
                            try:
                                cd = response.headers.get('Content-Disposition')           
                                filename = self._get_filename_from_cd(cd)                   
                            except Exception as e:
                                print(f"Failed to extract filename from Content-Disposition: {e}")
                            if not filename:
                                filename = self._get_filename_from_file_url(file_url)
                            if not filename:
                                filename = os.path.basename(urllib.parse.urlparse(file_url).path)
                            sanitized_filename = self._sanitize_filename(filename)
                            file_path = os.path.join(room_dir, sanitized_filename)
                            # Save the attachment
                            with open(file_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=1024):
                                    f.write(chunk)
                            message_data['files'].append(sanitized_filename)
                        except Exception as e:
                            print(f"Failed to download file: {e}")
                messages_data.append(message_data)

            # Prepare for next iteration
            last_message_id = messages_page[-1].id  # ID of the last message in the current page

            # If the number of messages retrieved is less than max_items, we've reached the last page
            if len(messages_page) < max_items:
                break

        # Save messages to JSON file
        with open(messages_file, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=4, default=str)
            
        # Generate HTML file
        self._generate_html_chat(messages_data, html_file)
    
    def export_direct_messages(self, output_dir):
        """
        Exports all direct (1-on-1) messages.
        Args:
            output_dir (str): The directory where the exported messages and attachments will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("\nExporting 1-on-1 chats...")
        direct_rooms = self.api.rooms.list(type='direct')
        for room in direct_rooms:
            self._export_room_messages(room, output_dir)

    def export_group_spaces(self, output_dir):
        """
        Exports all group spaces (rooms not part of a team).
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("\nExporting group spaces...")
        group_rooms = self.api.rooms.list(type='group')
        for room in group_rooms:
            # Exclude team spaces (handled separately)
            if not room.teamId:
                self._export_room_messages(room, output_dir)

    def export_team_spaces(self, output_dir):
        """
        Exports all teams and their associated spaces.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("\nExporting teams...")
        teams = self.api.teams.list()
        for team in teams:
            team_dir = os.path.join(output_dir, self._sanitize_filename(team.name))
            if not os.path.exists(team_dir):
                os.makedirs(team_dir)
            print(f"\nExporting team: {team.name}")
            team_rooms = self.api.rooms.list(teamId=team.id)
            for room in team_rooms:
                self._export_room_messages(room, team_dir)
