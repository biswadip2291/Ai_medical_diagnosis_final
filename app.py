import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import json

# --- Page Configuration ---
st.set_page_config(page_title="AI Medical Consultant", page_icon="🧑‍⚕️", layout="wide")

# --- Load LLM Model ---
@st.cache_resource
def load_llm():
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        llm = genai.GenerativeModel('gemini-1.5-flash-latest')
        return llm
    except KeyError:
        st.error("🚨 Google API Key not found! Please add it to your Streamlit secrets.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during model loading: {e}")
        st.stop()

model = load_llm()

# --- Core API Functions ---
def get_gemini_json_response(prompt, image=None):
    """Gets a JSON response from Gemini, for generating questions."""
    content = [prompt, image] if image else [prompt]
    try:
        response = model.generate_content(content)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return cleaned_response
    except Exception as e:
        return f'{{"error": "API communication failed: {e}"}}'

def get_gemini_text_response(prompt, image=None):
    """Gets a standard text/markdown response from Gemini, for the final analysis."""
    content = [prompt, image] if image else [prompt]
    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"Error: API communication failed. {e}"

# --- Prompt Engineering Functions ---
def create_triage_prompt():
    """Prompt 1: Asks the AI to generate triage questions based on the image."""
    return """
    You are an AI Triage Assistant. Look at the provided medical image and generate 2-3 important, multiple-choice questions to ask the user for context.
    Return a response in a strict JSON format with a single key "questions".
    The value of "questions" should be a list of objects, where each object has "question_text" and a list of "options".
    Example: {"questions": [{"question_text": "How long has this been present?", "options": ["< 1 day", "1-3 days", "> 3 days"]}]}
    Respond ONLY with the JSON object.
    """

def create_final_analysis_prompt(conversation_history):
    """
    Prompt 2: Asks for a final analysis where each interpretation is a
    separate, self-contained paragraph.
    """
    history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history])

    return f"""
    You are an expert medical analyst AI. You will be provided with a medical image and a triage conversation transcript.
    Your task is to provide a comprehensive, safe, and transparent final analysis based on BOTH the image and the conversation.

    **Triage Conversation Transcript:**
    {history_str}

    **Analysis Task:**
    Provide a detailed analysis formatted with Markdown, including these exact sections:
    
    1.  **Integrated Observation:** A brief summary combining visual findings with user-provided symptoms.
    
    2.  **Key Visual Characteristics:** A bulleted list of objective visual details.

    3.  **Potential Interpretation (Multi-Paragraph Format):**
        - In this section, discuss the most likely interpretations in **separate paragraphs**.
        - **First Paragraph:** Begin with the most likely possibility. State your confidence level (e.g., "Confidence: High") and then, in a narrative style, explain the supporting visual and user evidence for this conclusion.
        - **Subsequent Paragraph(s):** In a new paragraph, discuss a less likely possibility. State its confidence (e.g., "Confidence: Low") and explain why it is less likely, referencing the available evidence.
        - Discuss no more than three possibilities in total.

    4.  **Crucial Next Steps & Safety Information:**
    
    5.  **MANDATORY DISCLAIMER:**

    Structure your response exactly as requested, with separate paragraphs for each interpretation.
    """

# --- Initialize Session State ---
if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False
if 'triage_questions' not in st.session_state:
    st.session_state.triage_questions = []
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'final_analysis' not in st.session_state:
    st.session_state.final_analysis = None
if 'image' not in st.session_state:
    st.session_state.image = None

# --- Streamlit App Interface ---
st.title("🧑‍⚕️ AI Medical Consultant")
st.markdown("An interactive AI that asks clarifying questions to provide a more personalized analysis.")
st.warning("**WARNING:** This is an educational proof-of-concept and NOT a medical diagnosis. Always consult a qualified healthcare professional.")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("Image & Triage Chat")
    uploaded_file = st.file_uploader("1. Upload an image to begin...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        if 'image' not in st.session_state or st.session_state.image is None:
             st.session_state.image = image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if not st.session_state.conversation_started and st.session_state.final_analysis is None:
            if st.button("2. Start Triage Conversation", type="primary"):
                with st.spinner("AI is assessing the image..."):
                    prompt = create_triage_prompt()
                    response_str = get_gemini_json_response(prompt, st.session_state.image)
                    try:
                        data = json.loads(response_str)
                        st.session_state.triage_questions = data.get("questions", [])
                        st.session_state.conversation_started = True
                        st.session_state.current_question_index = 0
                        st.session_state.conversation_history = []
                        st.rerun()
                    except (json.JSONDecodeError, KeyError):
                        st.error("The AI failed to generate valid triage questions. Please try again.")

        if st.session_state.conversation_started:
            for q, a in st.session_state.conversation_history:
                with st.chat_message("assistant"):
                    st.write(q)
                with st.chat_message("user"):
                    st.write(a)
            
            if st.session_state.current_question_index < len(st.session_state.triage_questions):
                current_q = st.session_state.triage_questions[st.session_state.current_question_index]
                q_text = current_q["question_text"]
                q_options = current_q["options"]

                with st.chat_message("assistant"):
                    st.write(q_text)
                    cols = st.columns(len(q_options))
                    for i, option in enumerate(q_options):
                        if cols[i].button(option, key=f"q{st.session_state.current_question_index}_{option}"):
                            st.session_state.conversation_history.append((q_text, option))
                            st.session_state.current_question_index += 1
                            st.rerun()
            else:
                st.session_state.conversation_started = False
                with st.spinner("Generating final analysis based on your feedback..."):
                    final_prompt = create_final_analysis_prompt(st.session_state.conversation_history)
                    final_analysis_text = get_gemini_text_response(final_prompt, st.session_state.image)
                    st.session_state.final_analysis = final_analysis_text
                    st.rerun()

with col2:
    st.header("Final AI Analysis")
    if st.session_state.final_analysis:
        st.markdown(st.session_state.final_analysis)
    else:
        st.info("The final, evidence-based analysis will appear here after you complete the triage conversation.")