# ============================================================
# EXPERIMENT 3
# TEXT SUMMARIZATION AND QUESTION ANSWERING
# Compatible with Latest Transformers
# ============================================================

!pip install -q transformers torch sentencepiece accelerate

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

print("=" * 60)
print("EXPERIMENT 3: NLP APPLICATIONS")
print("=" * 60)

# ============================================================
# PART A : TEXT SUMMARIZATION USING BART
# ============================================================

print("\nLoading BART Summarization Model...")

summarizer_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/bart-large-cnn"
)

summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/bart-large-cnn"
)

text = """
Artificial Intelligence is transforming many industries by enabling
machines to perform tasks that normally require human intelligence.
It is widely used in healthcare, education, manufacturing, finance,
transportation, and cybersecurity. AI systems can analyze large
amounts of data, identify patterns, make predictions, and support
intelligent decision-making. Generative AI is a branch of Artificial
Intelligence that can create new content such as text, images, audio,
video, and computer programs.
"""

inputs = summarizer_tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=1024
)

summary_ids = summarizer_model.generate(
    inputs["input_ids"],
    max_length=60,
    min_length=20,
    num_beams=4,
    early_stopping=True
)

summary = summarizer_tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("\n" + "=" * 60)
print("PART A : TEXT SUMMARIZATION")
print("=" * 60)

print("\nOriginal Text:")
print(text)

print("\nGenerated Summary:")
print(summary)

# ============================================================
# PART B : QUESTION ANSWERING
# ============================================================

print("\nLoading Question Answering Model...")

qa_pipeline = pipeline(
    task="document-question-answering",
    model="distilbert-base-cased-distilled-squad"
)

context = """
Generative Artificial Intelligence is a type of Artificial Intelligence
that can create new content such as text, images, audio, video, and
computer programs. Large Language Models are commonly used for text
generation, summarization, translation, and question answering.
"""

question = "What type of content can Generative AI create?"

result = qa_pipeline(
    question=question,
    context=context
)

print("\n" + "=" * 60)
print("PART B : QUESTION ANSWERING")
print("=" * 60)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(result["answer"])

print("\nConfidence Score:")
print(round(result["score"], 3))

print("\nExperiment Completed Successfully!")
