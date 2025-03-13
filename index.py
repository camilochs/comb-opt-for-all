import streamlit as st
from openai import OpenAI
import os
import requests
from streamlit_modal import Modal

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Optimization Chat Assistant", layout="wide")
st.title("Improving Existing Optimization Algorithms with LLMs")

# --- ALERTA INICIAL ---
st.warning(
    "Important: Please make sure to complete the following elements to initiate the conversation: "
    "upload a file, select at least one checkbox, choose the function to improve, and provide an API key."
)

# --- API KEY CONFIGURATION ---
st.sidebar.header("API Key Configuration")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
huggingface_key = st.sidebar.text_input("Hugging Face API Key", type="password")

# Configure OpenAI client if API key is provided
client = None
if openai_key:
    client = OpenAI(api_key=openai_key)
elif "openai_api_key" in st.secrets:
    client = OpenAI(api_key=st.secrets["openai_api_key"])
elif os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- SESSION STATE INITIALIZATION ---
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
    # Agregar mensaje del sistema para definir el rol del asistente
    st.session_state.conversation_history.append({
        "role": "system",
        "content": "I'm an expert in optimization algorithms."
    })
    # Mensaje de bienvenida del asistente
    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": "Hello! I'm your Optimization Assistant. How can I help you today?"
    })

# --- SIDEBAR: CONFIGURACIÓN PARA EL PROMPT ---
st.sidebar.header("Configuration")
function_name = st.sidebar.text_input("Which function needs improvement?")
st.sidebar.subheader("What to improve?")
option1 = st.sidebar.checkbox("Heuristic", value=True)
option2 = st.sidebar.checkbox("C++ code")
uploaded_file = st.sidebar.file_uploader("Upload your optimization algorithm (e.g., main.cpp)", type=["cpp", "c"])
file_content = ""
if uploaded_file is not None:
    file_content = uploaded_file.getvalue().decode("utf-8")

# --- ELEMENTO PARA MODIFICAR LA TEMPERATURA ---
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

def generate_prompt(user_input, option1, option2, file_content, function_name):
    prompt = f"{user_input}"
    
    if file_content:
        prompt += f"\n\nCode of CMSA for MIS:\n\n{file_content}"

    if option1:
        prompt += f"\n\n- Analyze the function `{function_name}` and identify a better heuristic to improve its performance."
    if option2:
        prompt += f"\n\n- Given the following heuristic implemented in C++ (`{function_name}`), without altering its core logic or functionality, please analyze the code to identify potential improvements in the use of data structures, cache optimization, and other low-level optimizations. Focus on enhancing performance by suggesting more efficient data structures, reducing memory overhead, improving data locality, and leveraging modern C++ features where applicable. Please provide an updated version of the code with comments explaining each optimization."
   
    prompt += f"\n\nPlease provide only the `{function_name}` function in C++ code, along with its justification. I do not need any additional C++ code or context."
    
    return prompt

def get_openai_response(prompt, model):
    try:
        # Incluir el historial completo de la conversación junto con el nuevo prompt del usuario
        messages = st.session_state.conversation_history + [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,  # Se utiliza la temperatura definida por el usuario
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API Error: {e}"

def get_huggingface_response(prompt, model):
    try:
        if not huggingface_key:
            return "Hugging Face API Key is missing."
        headers = {"Authorization": f"Bearer {huggingface_key}"}
        payload = {"inputs": prompt}
        response = requests.post(f"https://api-inference.huggingface.co/models/{model}",
                                 headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return f"Hugging Face API Error: {response.text}"
    except Exception as e:
        return f"Hugging Face API Error: {e}"

# --- MODEL SELECTION ---
st.sidebar.header("Select AI Model")
model_provider = st.sidebar.radio("Choose API:", ["OpenAI", "Hugging Face"])
if model_provider == "OpenAI":
    model = st.sidebar.selectbox("OpenAI Model:", ["chatgpt-4o-latest"])
else:
    model = st.sidebar.selectbox("Hugging Face Model:", [
        "deepseek-ai/DeepSeek-R1",
        "Qwen/Qwen2.5-14B-Instruct-1M",
        "meta-llama/Llama-3.3-70B-Instruct",
        "mistralai/Codestral-22B-v0.1",
    ])

# --- BOTÓN PARA RESETEAR LA CONVERSACIÓN ---
if st.sidebar.button("Reset Conversation"):
    st.session_state.conversation_history = []
    st.session_state.conversation_history.append({
        "role": "system",
        "content": "I'm an expert in optimization algorithms."
    })
    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": "Hello! I'm your Optimization Assistant. How can I help you today?"
    })
    st.rerun()

st.header("Chat Conversation")

# Mostrar el historial de la conversación usando los componentes de chat_message
for msg in st.session_state.conversation_history:
    if msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    elif msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "system":
        st.info(f"System: {msg['content']}")

if "first_time" not in st.session_state:
    st.session_state.first_time = False

user_followup = st.chat_input(
    "Here is the CMSA algorithm in C++ for solving the Maximum Independent Set (MIS) problem. Please consider the 'age' parameter of CMSA when designing the new heuristic.", 
    key="initial_input"
)
if user_followup:
    if not st.session_state.first_time:
        prompt = generate_prompt(user_followup, option1, option2, file_content, function_name)
        st.session_state.first_time = True
    else:
        prompt = user_followup

    # Mostrar el prompt antes de la llamada al modelo
    st.markdown("### Prompt:")
    st.code(prompt, language="text")
    
    with st.spinner("Consulting the model..."):
        if model_provider == "OpenAI":
            followup_response = get_openai_response(prompt, model)
        else:
            followup_response = get_huggingface_response(prompt, model)
    
    # Agregar el prompt y la respuesta a la conversación
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append({"role": "assistant", "content": followup_response})
    st.rerun()
