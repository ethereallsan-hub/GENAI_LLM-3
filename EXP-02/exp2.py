# ============================================================
# EXPERIMENT 2
# SENTIMENT ANALYSIS AND DOCUMENT CLASSIFICATION
# Google Colab Compatible
# ============================================================

!pip install -q transformers torch accelerate sentencepiece

from transformers import pipeline

print("=" * 60)
print("EXPERIMENT 2: TEXT CLASSIFICATION")
print("=" * 60)

# ============================================================
# PART A : SENTIMENT ANALYSIS
# ============================================================

print("\nLoading Sentiment Analysis Model...")

sentiment_analyzer = pipeline(
    task="sentiment-analysis"
)

text = "The Generative AI workshop was extremely informative and useful."

result = sentiment_analyzer(text)

print("\n" + "=" * 60)
print("PART A : SENTIMENT ANALYSIS")
print("=" * 60)

print("\nInput Text:")
print(text)

print("\nPredicted Sentiment:")
print(result)

# ============================================================
# PART B : DOCUMENT CLASSIFICATION
# ============================================================

print("\nLoading Zero-Shot Classification Model...")

classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli"
)

document = """
Artificial Intelligence and Machine Learning are transforming
industries through automation and intelligent decision-making.
"""

labels = [
    "Technology",
    "Sports",
    "Politics",
    "Entertainment"
]

result = classifier(
    document,
    labels
)

print("\n" + "=" * 60)
print("PART B : DOCUMENT CLASSIFICATION")
print("=" * 60)

print("\nDocument:")
print(document)

print("\nPredicted Labels:")
for label, score in zip(result["labels"], result["scores"]):
    print(f"{label}: {score:.4f}")

print("\nMost Relevant Category:")
print(result["labels"][0])

print("\nExperiment Completed Successfully!")
