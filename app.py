import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import gradio as gr

# ✅ Load model and tokenizer from your folder
model_path = "sentiments_model"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# Label mapping — change based on your training
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

def classify_tweet(tweet):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    return f"{id2label[predicted_class]} ({confidence:.2%} confidence)"

# ✅ Create Gradio interface
iface = gr.Interface(fn=classify_tweet, 
                     inputs=gr.Textbox(lines=2, placeholder="Enter a tweet here..."),
                     outputs="text",
                     title="Tweet Sentiment Classifier")

iface.launch()
