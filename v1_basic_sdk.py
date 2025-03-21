import openai
import os
import re
import requests
import time
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch_website_content(url):
    """Fetch and parse content from a website URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text (remove excessive whitespace)
        clean_text = re.sub(r'\s+', ' ', text).strip()
        
        return clean_text
    except Exception as e:
        return f"Error fetching website content: {str(e)}"

def create_business_analyst_assistant():
    """Create and return a business analyst assistant."""
    assistant = client.beta.assistants.create(
        name="Business Analyst Assistant",
        instructions="""
        You are a specialized business analyst assistant. Your task is to analyze company information and provide a structured report.
        You must provide a comprehensive analysis with the following sections:
        1. Company Summary: Brief overview of what the company does
        2. Key Features/Products: Main offerings or products
        3. Purpose/Mission: The company's mission and goals
        4. Main Competitors: List of major competitors in the same space
        5. Target Market: Who the company is serving
        
        Format the response in a clean, structured way that's easy to read.
        """,
        model="gpt-4-turbo",
        tools=[{"type": "retrieval"}]  # Enable knowledge retrieval
    )
    return assistant

def analyze_company_with_assistant(input_text):
    """
    Analyze a company using the Assistants API.
    
    Args:
        input_text: Company name or website URL
        
    Returns:
        String containing company analysis
    """
    # Check if input is a URL
    is_url = input_text.startswith(('http://', 'https://'))
    
    # If it's a URL, fetch the content
    if is_url:
        website_content = fetch_website_content(input_text)
        company_identifier = input_text
        message_content = f"""
        Analyze the following company based on its website content:
        URL: {company_identifier}
        
        Website Content:
        {website_content[:4000]}  # Limiting content length
        
        Please provide a detailed analysis of the company.
        """
    else:
        company_identifier = input_text
        message_content = f"""
        Analyze the following company:
        Company Name: {company_identifier}
        
        Please provide a detailed analysis of the company based on your knowledge.
        """
    
    # Create or get the assistant
    assistant = create_business_analyst_assistant()
    
    # Create a new thread for this analysis
    thread = client.beta.threads.create()
    
    # Add the user message to the thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message_content
    )
    
    # Run the assistant on the thread
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    
    # Poll for the run to complete
    while run.status in ["queued", "in_progress"]:
        print(".", end="", flush=True)
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
    
    print()  # New line after progress dots
    
    # Get the messages after the run completes
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    
    # Find the assistant's response (the most recent assistant message)
    assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
    if assistant_messages:
        # Get the latest assistant message
        latest_message = assistant_messages[0]
        # Extract the text content from the message
        if latest_message.content:
            content_parts = latest_message.content[0].text
            return content_parts.value
    
    return "No analysis was generated."

def main():
    print("OpenAI SDK Research Assistant (Using Assistants API)")
    print("=================================================")
    print("Enter a company name or website URL to analyze:")
    
    while True:
        user_input = input("\nCompany name or URL (or 'quit' to exit): ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_input.strip():
            print("Please enter a valid company name or URL.")
            continue
            
        print("\nAnalyzing... This may take a moment.")
        
        try:
            analysis = analyze_company_with_assistant(user_input)
            print("\n" + analysis)
        except Exception as e:
            print(f"Error performing analysis: {str(e)}")
    
    print("\nThank you for using the Research Assistant!")

if __name__ == "__main__":
    main()
