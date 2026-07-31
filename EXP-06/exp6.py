!pip install transformers sentence-transformers faiss-cpu sentencepiece -q

import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Knowledge Base
# -----------------------------
documents = [
    """Generative Artificial Intelligence is a branch of AI that creates
    new content such as text, images, audio, video and computer programs.""",

    """Large Language Models are transformer-based models trained on massive
    text datasets. They are used for text generation, summarization,
    translation, question answering and conversational AI.""",

    """Retrieval-Augmented Generation combines information retrieval with
    text generation. It retrieves relevant documents from an external
    knowledge base and gives them to a language model as context.""",

    """Vector databases store high-dimensional embeddings and perform
    similarity searches. Examples include FAISS, ChromaDB, Pinecone,
    Weaviate and Milvus.""",

    """Prompt engineering is the process of designing clear instructions
    that guide a language model to produce accurate and useful responses.""",

    """Fine-tuning adapts a pretrained language model to a specific domain
    or task using a smaller domain-specific dataset."""
]

# -----------------------------
# Embedding Model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
).astype("float32")

faiss.normalize_L2(document_embeddings)

# -----------------------------
# FAISS Vector Database
# -----------------------------
dimension = document_embeddings.shape[1]

vector_database = faiss.IndexFlatIP(dimension)

vector_database.add(document_embeddings)

# -----------------------------
# FLAN-T5 Model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# -----------------------------
# Retrieval Function
# -----------------------------
def retrieve_documents(query, top_k=2):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = vector_database.search(
        query_embedding,
        top_k
    )

    results = []

    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "document": documents[idx],
            "score": float(score)
        })

    return results

# -----------------------------
# Generation Function
# -----------------------------
def generate_answer(query, retrieved_documents):

    context = "\n\n".join(
        item["document"]
        for item in retrieved_documents
    )

    prompt = f"""
Answer the question using ONLY the information provided below.

Context:
{context}

Question:
{query}

If the answer is not found in the context,
respond with:
"The answer is not available in the knowledge base."
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer

# -----------------------------
# Main Program
# -----------------------------
print("=" * 50)
print("RETRIEVAL-AUGMENTED GENERATION SYSTEM")
print("=" * 50)

query = input("\nEnter your question: ")

retrieved_docs = retrieve_documents(
    query=query,
    top_k=2
)

answer = generate_answer(
    query=query,
    retrieved_documents=retrieved_docs
)

print("\nRETRIEVED DOCUMENTS")
print("-" * 50)

for i, item in enumerate(retrieved_docs, start=1):
    print(f"\nDocument {i}")
    print(item["document"])
    print(f"Similarity Score: {item['score']:.4f}")

print("\nGENERATED ANSWER")
print("-" * 50)
print(answer)
