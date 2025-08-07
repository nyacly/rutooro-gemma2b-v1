import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- 1. Load Model and Tokenizer ---
# Use 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the fine-tuned model and tokenizer from the Hugging Face Hub
model_id = "cle-13/gemma-2b-it-rutooro-A100"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# --- 2. Define the Chat Logic ---
def get_response(message, history):
    """
    This function processes the user message and chat history to generate a model response.
    """
    # Convert the chat history from Gradio's format to the list of dictionaries
    # that the chat template expects.
    chat_history_for_template = []
    for user_msg, assistant_msg in history:
        chat_history_for_template.append({"role": "user", "content": user_msg})
        chat_history_for_template.append({"role": "assistant", "content": assistant_msg})

    # Add the user's latest message
    chat_history_for_template.append({"role": "user", "content": message})

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(
        chat_history_for_template,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the prompt and move it to the model's device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate a response
    # The streamer is used to yield tokens as they are generated
    # for a more responsive UI.
    streamer = gr.OAuthTokenStreamer()

    # We need to run the generation in a separate thread to avoid blocking the UI
    def generate():
        with torch.no_grad():
            model.generate(**inputs, streamer=streamer, max_new_tokens=256, do_sample=True, top_p=0.95, top_k=50)

    # Start generation in a separate thread
    thread = torch.multiprocessing.Process(target=generate)
    thread.start()

    # Yield tokens as they become available
    for new_text in streamer:
        yield new_text

# --- 3. Build and Launch the Gradio Interface ---
# Define the title and description for the Gradio app
APP_TITLE = "Rutooro AI Chat"
APP_DESCRIPTION = """
This is a chat interface for a Gemma 2B model fine-tuned for the Rutooro language.
You can chat with it in Rutooro or ask it to perform translations.
"""

# Create the Gradio ChatInterface
chatbot = gr.ChatInterface(
    fn=get_response,
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    examples=[
        ["Orahire ota?"],
        ["Translate 'I am learning to code' to Rutooro"],
        ["Ninyenda kwega okucoda"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    chatbot.launch()
