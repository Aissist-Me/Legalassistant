import asyncio
import base64
import json
import traceback
import nest_asyncio
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import streamlit as st

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

openai_api_key = st.secrets["api_keys"]["openai_api_key"]
proxy_api_key = st.secrets["api_keys"]["proxy_api_key"]

# Initialize the OpenAI client with the secure API key
client = OpenAI(api_key=openai_api_key)

# Global thread initialization
if 'user_thread' not in st.session_state:
    st.session_state.user_thread = client.beta.threads.create()

# Proxy setup
PROXY_URL = 'https://proxy.scrapeops.io/v1/'
API_KEY = proxy_api_key

def scrape_content(url):
    """Fetches HTML from the target URL using the proxy service, extracts text content, and deduplicates href links."""
    params = {
        'api_key': API_KEY,
        'url': url,
        'render_js': 'true',
        'residential': 'true',
    }
    
    try:
        with st.spinner(f"Scraping content from {url}..."):
            # Make the request to the proxy service
            response = requests.get(PROXY_URL, params=params)
            response.encoding = 'utf-8'  # Enforce UTF-8 encoding
            
            # Check if the request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract and clean the text content
                content = soup.get_text(separator="\n", strip=True)
                
                # Extract and deduplicate href links using set comprehension
                links = sorted({a.get('href') for a in soup.find_all('a', href=True) if a.get('href')})
                
                st.success(f"Successfully scraped content from {url}")
                return {
                    'content': content,
                    'links': links  # Return the sorted list of links
                }
            else:
                st.error(f"Failed to fetch the page: {url}, status code: {response.status_code}")
                return None
    except requests.exceptions.Timeout:
        st.error(f"Request timed out while trying to scrape {url}.")
        return None
    except requests.exceptions.TooManyRedirects:
        st.error(f"Too many redirects while trying to scrape {url}.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while scraping {url}: {e}")
        return None

# Define function specifications for content scraping
tools = [
    {
        "type": "function",
        "function": {
            "name": "scrape_content",
            "description": "Use this function to scrape text content from a given URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to scrape content from."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {"type": "code_interpreter"},
        {"type": "file_search"}

]

available_functions = {
    "scrape_content": scrape_content,
}


# Instructions for the assistant
instructions = """
You are a Legal Document Assistant specializing in legal research and answering user queries based on documents like PDF files and legal texts.
You have access to both internal documents and a web scraping function for additional information if necessary. Here’s how you should operate:

Key Functions:

Document Search and Analysis: Search through internal PDF files and documents for relevant sections before offering any external sources. 
Always ensure responses reference the jurisdiction of Jamaica as outlined in the documents.

Scraping Feature: If no sufficient information is found internally, or if the user requests more, you can suggest web scraping as a last resort, but only 
with the user's explicit approval.

Legal Research and Contextual Explanation: Provide detailed explanations of laws, statutes, and regulations based on the content of the documents, 
specifically related to Jamaica.

Structured and Clear Responses: Offer structured and clear answers, referencing specific sections of the law for transparency.

Compensation Calculation: Help the user calculate fines, penalties, or compensation based on the laws in the documents.

Interaction Flow:

Document First Approach: Always begin by searching through the provided legal documents. If the information is found, respond by quoting the specific sections.

Ask for Web Scraping Permission: If the relevant information is not found within the documents, ask the user if they would like you to scrape external sources.

User Query Follow-Up: Always ensure the user has all the legal context and ask if they need further clarification or related information.

Clarification and Refinement: Continuously refine your responses based on feedback from the user to provide tailored legal advice.

Example Interaction:

User Input: "What are the penalties for unauthorized access to computer data under the Cybercrimes Act?"

Assistant Response: "According to Section 3 of the Cybercrimes Act, unauthorized access to computer data is an offense. For a first offense, upon summary 
conviction, you may be fined up to three million dollars or face up to three years of imprisonment. 
This applies under Jamaican jurisdiction. Shall I search for more cases or legal precedents related to this?"

"""

# Function to create an assistant
def create_assistant(file_ids):
    assistant = client.beta.assistants.create(
        name="LegalAI assistant",
        instructions=instructions,
        model="gpt-4o-mini",
        tools=tools,
        tool_resources={
            'file_search': {
                'vector_stores': [{
                    'file_ids': file_ids
                }]
            }
        }
    )
    return assistant.id

def safe_tool_call(func, tool_name, **kwargs):
    """Safely execute a tool call and handle exceptions."""
    try:
        result = func(**kwargs)
        return result if result is not None else f"No content returned from {tool_name}"
    except Exception as e:
        st.error(f"Error in {tool_name}: {str(e)}")
        return f"Error occurred in {tool_name}: {str(e)}"
    
    # Function to handle tool outputs
def handle_tool_outputs(run):
    tool_outputs = []
    try:
        for call in run.required_action.submit_tool_outputs.tool_calls:
            function_name = call.function.name
            function = available_functions.get(function_name)
            if not function:
                raise ValueError(f"Function {function_name} not found in available_functions.")
            arguments = json.loads(call.function.arguments)
            # Use safe_tool_call if necessary
            with st.spinner(f"Executing a detailed search..."):
                output = safe_tool_call(function, function_name, **arguments)

            tool_outputs.append({
                "tool_call_id": call.id,
                "output": json.dumps(output)
            })

        # Use the correct user-specific thread ID here
        return client.beta.threads.runs.submit_tool_outputs(
            thread_id=st.session_state.user_thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )
    except Exception as e:
        st.error(f"Error in handle_tool_outputs: {str(e)}")
        st.error(traceback.format_exc())
        return None


# Function to get agent response
async def get_agent_response(assistant_id, user_message):
    try:
        with st.spinner("Processing your request..."):
            # Use the unique thread for each session
            client.beta.threads.messages.create(
                thread_id=st.session_state.user_thread.id,
                role="user",
                content=user_message,
            )

            run = client.beta.threads.runs.create(
                thread_id=st.session_state.user_thread.id,
                assistant_id=assistant_id,
            )

            while run.status in ["queued", "in_progress"]:
                run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.user_thread.id,
                    run_id=run.id
                )
                if run.status == "requires_action":
                    run = handle_tool_outputs(run)
                await asyncio.sleep(1)

            last_message = client.beta.threads.messages.list(thread_id=st.session_state.user_thread.id, limit=1).data[0]

            formatted_response_text = ""
            download_links = []
            images = []

            if last_message.role == "assistant":
                for content in last_message.content:
                    if content.type == "text":
                        formatted_response_text += content.text.value
                        for annotation in content.text.annotations:
                            if annotation.type == "file_path":
                                file_id = annotation.file_path.file_id
                                file_name = annotation.text.split('/')[-1]
                                file_content = client.files.content(file_id).read()
                                download_links.append((file_name, file_content))
                    elif content.type == "image_file":
                        file_id = content.image_file.file_id
                        image_data = client.files.content(file_id).read()
                        images.append((f"{file_id}.png", image_data))
                        formatted_response_text += f"[Image generated: {file_id}.png]\n"
            else:
                formatted_response_text = "Error: No assistant response"

            return formatted_response_text, download_links, images
    except Exception as e:
        st.error(f"Error in get_agent_response: {str(e)}")
        st.error(traceback.format_exc())
        return f"Error: {str(e)}", [], []

# Streamlit app
def main():
    st.title("Legal assistant")

    # Sidebar for assistant selection
    st.sidebar.title("Assistant Configuration")
    assistant_choice = st.sidebar.radio("Choose an option:", ["Create New Assistant", "Use Existing Assistant"])

    if assistant_choice == "Create New Assistant":
        # File uploader
        uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
        file_ids = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_info = client.files.create(file=uploaded_file, purpose='assistants')
                file_ids.append(file_info.id)

        if file_ids:
            if st.sidebar.button("Create New Assistant"):
                st.session_state.assistant_id = create_assistant(file_ids)
                st.sidebar.success(f"New assistant created with ID: {st.session_state.assistant_id}")
        else:
            st.sidebar.warning("Please upload files to create an assistant.")

    else:  # Use Existing Assistant
        assistant_id = st.sidebar.text_input("Enter existing assistant ID:")
        if assistant_id:
            st.session_state.assistant_id = assistant_id
            st.sidebar.success(f"Using assistant with ID: {assistant_id}")

    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "downloads" in message:
                for file_name, file_content in message["downloads"]:
                    st.download_button(
                        label=f"Download {file_name}",
                        data=file_content,
                        file_name=file_name,
                        mime="application/octet-stream"
                    )
                    if file_name.endswith('.html'):
                        st.components.v1.html(file_content.decode(), height=300, scrolling=True)
            if "images" in message:
                for image_name, image_data in message["images"]:
                    st.image(image_data)
                    st.download_button(
                        label=f"Download {image_name}",
                        data=image_data,
                        file_name=image_name,
                        mime="image/png"
                    )

    if prompt := st.chat_input("You:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if 'assistant_id' in st.session_state:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response, download_links, images = asyncio.run(get_agent_response(st.session_state.assistant_id, prompt))
                message_placeholder.markdown(response)
                
                for file_name, file_content in download_links:
                    st.download_button(
                        label=f"Download {file_name}",
                        data=file_content,
                        file_name=file_name,
                        mime="application/octet-stream"
                    )
                    if file_name.endswith('.html'):
                        st.components.v1.html(file_content.decode(), height=300, scrolling=True)
                
                for image_name, image_data in images:
                    st.image(image_data)
                    st.download_button(
                        label=f"Download {image_name}",
                        data=image_data,
                        file_name=image_name,
                        mime="image/png"
                    )
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response, 
                "downloads": download_links,
                "images": images
            })
        else:
            st.warning("Please create a new assistant or enter an existing assistant ID before chatting.")

if __name__ == "__main__":
    main()
