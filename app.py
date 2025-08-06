from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import gradio as gr

# ✅ Make sure this folder name matches exactly
model_path = "sentiment_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Optional label mapping (update if your model has different labels)
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

def classify_tweet(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
    return f"{id2label[predicted]} ({confidence:.2%} confidence)"

# Gradio interface
gr.Interface(
    fn=classify_tweet,
    inputs=gr.Textbox(lines=3, placeholder="Enter a tweet..."),
    outputs="text",
    title="Tweet Sentiment Classifier"
).launch()
