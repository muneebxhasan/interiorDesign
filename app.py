import streamlit as st
from typing import List
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
import base64
from PIL import Image
from io import BytesIO
# from dotenv import find_dotenv, load_dotenv
import os
import requests
from langchain_openai import ChatOpenAI

# Load environment variables and configure page
st.set_page_config(page_title="Interior Design Consultation", layout="wide")
# load_dotenv(find_dotenv())



# os.getenv("GITHUB_TOKEN") == st.secrets["GITHUB_TOKEN"]
# os.getenv("NVIDIA_2") == st.secrets["NVIDIA_2"]
# NVIDIA_API_KEY = os.getenv("NVIDIA_2")
# if not API_KEY or not NVIDIA_API_KEY:
#     st.error("Missing required environment variables. Please check your .env file.")
#     st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "completed" not in st.session_state:
    st.session_state.completed = False
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Constants
DESIGN_QUESTIONS = [
    "What would you like to see in your area?",
    "What mood or atmosphere do you want to create in your space?",
    "What kind of ambience do you want to offer to others (e.g., visitors, customers, public)?",
    "What areas are you wanting to create within this space?",
    "What colors are you drawn to?",
    "Would you like assistance with selecting the right colors for your space?",
    "Where do you envision this space evolving in the next 2, 5, or 10 years?",
    "What are your main design goals?",
    "Are there any specific features you would like included?"
]

TOTAL_QUESTIONS = len(DESIGN_QUESTIONS)

# System prompt from original code
SYSTEM_PROMPT = f"""You are Ariel Clemens, a professional interior designer known for creating stunning, personalized spaces. 
Be conversational and engaging, showing genuine interest in their responses.
Provide thoughtful insights and suggestions based on their answers.Design questions {DESIGN_QUESTIONS}"""

# Initial greeting from original code
INITIAL_GREETING = """Hello! I'm Ariel , and I'm excited to help bring your interior design vision to life! 
Through our conversation, I'll ask you several questions to understand your needs and preferences better. 
Let's start with your overall vision. What would you like to see in your area?"""

# Initialize LLM
def initialize_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=st.secrets["GITHUB_TOKEN"],
        base_url="https://models.inference.ai.azure.com"
    )

# Styling
# st.markdown("""
#     <style>
#     /* Reset the input background to white and ensure text is visible */
#     .stTextInput > div > div > input {
#         background-color: black !important;
#         color: #262730 !important;
#         border: 1px solid #ddd !important;
#     }
    
#     /* Add hover effect for better interaction */
#     .stTextInput > div > div > input:hover {
#         border-color: #999 !important;
#     }
    
#     /* Add focus effect */
#     .stTextInput > div > div > input:focus {
#         border-color: #1f77b4 !important;
#         box-shadow: 0 0 0 1px #1f77b4 !important;
#     }

#     .chat-message {
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 1rem;
#         display: flex;
#         flex-direction: column;
#     }
#     .ai-message {
#         background-color: #f0f2f6;
#     }
#     .user-message {
#         background-color: #e1e9ff;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# Main title
st.title("🎨 Interior Design Consultation with Ariel")

# Initialize chat with system message and greeting
if not st.session_state.initialized:
    st.session_state.messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        AIMessage(content=INITIAL_GREETING)
    ]
    st.session_state.initialized = True

# Progress bar
progress = st.progress(0)
st.markdown(f"Question {st.session_state.current_question + 1} of {TOTAL_QUESTIONS}")

def generate_image(prompt: str):
    """Generate image using NVIDIA API"""
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl"
    
    headers = {
        "Authorization": f"Bearer {st.secrets["NVIDIA_2"]}",
        "Accept": "application/json",
    }
    
    payload = {
        "text_prompts": [
            {
                "text": prompt,
                "weight": 1
            },
            {
                "text": "",
                "weight": -1
            }
        ],
        "cfg_scale": 5,
        "sampler": "K_DPM_2_ANCESTRAL",
        "seed": 0,
        "steps": 25
    }
    
    response = requests.post(invoke_url, headers=headers, json=payload)
    response.raise_for_status()
    response_body = response.json()
    base64_string = response_body['artifacts'][0]['base64']
    return base64_string

def process_response(user_input: str):
    """Process user response and update state"""
    llm = initialize_llm()
    
    # Add user message to state
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Check if we've reached the end
    if st.session_state.current_question >= TOTAL_QUESTIONS - 1:
        # Generate final summary using original prompt
        final_prompt = SystemMessage(content="Provide a comprehensive summary on the design's vision. NO MARKDOWN, RESPONSE ONLY IN STRING FORMAT")
        final_response = llm.invoke(st.session_state.messages + [final_prompt])
        st.session_state.messages.append(AIMessage(content=final_response.content))
        
        # Generate image prompt using original format
        image_prompt = SystemMessage(content="""NO MARKDOWN, RESPONSE ONLY IN STRING FORMAT
                                    Based on the conversation history, create a detailed image generation prompt. 
        Include:
        1. Overall style and atmosphere
        2. Color scheme and materials
        3. Key features and layout
        4. Lighting and mood
        5. Specific design elements mentioned
        Format the response as a cohesive, detailed prompt suitable for image generation.""")
        prompt_response = llm.invoke(st.session_state.messages + [image_prompt])
        
        # Add technical specifications as in original code
        final_prompt = f"""Interior Design Visualization:
        {prompt_response.content}

        Technical Specifications:
        - Photorealistic rendering
        - Wide-angle perspective
        - Professional interior lighting
        - High-resolution 8K
        - Architectural details visible
        - Professional interior photography style"""
        
        # Generate and display image
        try:
            image_base64 = generate_image(final_prompt)
            st.session_state.final_image = image_base64
            st.session_state.completed = True
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
    else:
        # Process response and ask next question
        next_question = DESIGN_QUESTIONS[st.session_state.current_question + 1]
        response = llm.invoke(
            st.session_state.messages + 
            [SystemMessage(content=f"Acknowledge the response briefly and ask this question next: {next_question}")]
        )
        st.session_state.messages.append(AIMessage(content=response.content))
        st.session_state.current_question += 1

# Chat interface
chat_container = st.container()

# Display chat messages
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        with chat_container:
            st.markdown(f'<div class="chat-message ai-message">👩‍🎨 Ariel: {msg.content}</div>', 
                       unsafe_allow_html=True)
    elif isinstance(msg, HumanMessage):
        if msg.content != SYSTEM_PROMPT:  # Don't display system prompt
            with chat_container:
                st.markdown(f'<div class="chat-message user-message">👤 You: {msg.content}</div>', 
                           unsafe_allow_html=True)

# Update progress
progress.progress((st.session_state.current_question) / TOTAL_QUESTIONS)

# Input field
if not st.session_state.completed:
    if user_input := st.chat_input("Your response:"):
        process_response(user_input)
        # st.experimental_rerun()
        st.rerun()

# Display final image if consultation is complete
if st.session_state.completed and hasattr(st.session_state, 'final_image'):
    st.markdown("### 🎨 Your Design Visualization")
    image_data = base64.b64decode(st.session_state.final_image)
    image = Image.open(BytesIO(image_data))
    st.image(image, caption="Generated Interior Design Visualization")
    
    if st.button("Start New Consultation"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()