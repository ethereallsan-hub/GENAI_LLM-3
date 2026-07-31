# ============================================================
# EXPERIMENT 4
# Conversational AI Chatbot using Microsoft DialoGPT
# Google Colab Compatible
# ============================================================

!pip install -q transformers torch accelerate

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------
# Load Pretrained Conversational Model
# ------------------------------------------------------------

model_name = "microsoft/DialoGPT-small"

print("Loading DialoGPT Model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model Loaded Successfully!")

# ------------------------------------------------------------
# Chatbot Interface
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CONVERSATIONAL AI CHATBOT")
print("=" * 60)
print("Chatbot: Hello! Type 'exit' to end the conversation.\n")

chat_history_ids = None

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\nChatbot: Goodbye! Have a nice day.")
        break

    # Encode user input
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt"
    )

    # Append conversation history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat(
            [chat_history_ids, new_input_ids],
            dim=-1
        )
    else:
        bot_input_ids = new_input_ids

    # Generate chatbot response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode only the new response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    print("Chatbot:", response)
