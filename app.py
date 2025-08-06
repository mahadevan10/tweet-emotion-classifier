import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import numpy as np

# Load your emotion classification model
# Replace 'your-username/your-model-name' with your actual model path on Hugging Face
# Or use a local path if you're uploading model files to the space
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"  # Example model - replace with yours

def load_model():
    """Load the emotion classification model"""
    try:
        # Option 1: If using Hugging Face Hub model
        classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            return_all_scores=True
        )
        return classifier
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to a working model for demo
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        return classifier

# Initialize the model
emotion_classifier = load_model()

def preprocess_tweet(tweet):
    """
    Preprocess tweet text for better classification
    """
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags (optional - keep if they add context)
    # tweet = re.sub(r'@\w+|#\w+', '', tweet)
    
    # Remove extra whitespace
    tweet = ' '.join(tweet.split())
    
    return tweet.strip()

def classify_emotion(tweet_text):
    """
    Classify the emotion of a tweet
    """
    try:
        if not tweet_text or not tweet_text.strip():
            return "Please enter a tweet to analyze! 🐦", {}
        
        # Preprocess the tweet
        processed_tweet = preprocess_tweet(tweet_text)
        
        if len(processed_tweet) < 3:
            return "Tweet too short to analyze. Please enter a longer tweet. 📝", {}
        
        # Get predictions
        results = emotion_classifier(processed_tweet)
        
        # Format results for display
        if isinstance(results, list) and len(results) > 0:
            emotions = results[0] if isinstance(results[0], list) else results
            
            # Create a formatted output
            output_text = "🎭 **Emotion Analysis Results:**\n\n"
            output_text += f"**Tweet:** {tweet_text[:100]}{'...' if len(tweet_text) > 100 else ''}\n\n"
            
            # Sort emotions by confidence
            sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
            
            # Create confidence dictionary for the plot
            confidence_dict = {}
            
            for i, emotion in enumerate(sorted_emotions):
                emotion_name = emotion['label'].title()
                confidence = emotion['score']
                confidence_dict[emotion_name] = confidence
                
                if i == 0:  # Highest confidence
                    output_text += f"🏆 **Primary Emotion: {emotion_name}** ({confidence:.2%})\n"
                else:
                    output_text += f"   {emotion_name}: {confidence:.2%}\n"
            
            return output_text, confidence_dict
        else:
            return "Error in classification. Please try again. ❌", {}
            
    except Exception as e:
        return f"Error occurred: {str(e)} ❌", {}

def analyze_tweet(tweet_text):
    """
    Main function called by Gradio interface
    """
    result_text, confidence_scores = classify_emotion(tweet_text)
    return result_text, confidence_scores

# Create example tweets for users to try
example_tweets = [
    "I just got accepted into my dream university! This is the best day of my life! 🎉",
    "Stuck in traffic again... this commute is killing me 😤",
    "My dog passed away today. I'm going to miss him so much 😢",
    "Can't believe they canceled my favorite show! So disappointed right now.",
    "Beautiful sunset tonight. Feeling grateful for moments like these 🌅",
    "Why do people have to be so rude? Having a terrible day.",
    "Absolutely terrified about the presentation tomorrow. What if I mess up?",
    "Nothing exciting ever happens in my life. Just another boring day."
]

# Create the Gradio interface
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Tweet Emotion Classifier") as interface:
        gr.HTML("<h1 style='text-align: center; color: #1DA1F2;'>🐦 Tweet Emotion Classifier 🎭</h1>")
        gr.HTML("<p style='text-align: center; font-size: 18px;'>Analyze the emotions in tweets using AI! Enter any tweet below and discover its emotional tone.</p>")
        
        with gr.Row():
            with gr.Column(scale=2):
                tweet_input = gr.Textbox(
                    label="📝 Enter Tweet",
                    placeholder="What's happening? Share a tweet to analyze its emotion...",
                    lines=4,
                    max_lines=6,
                    info="Enter the tweet text you want to analyze (up to 280 characters recommended)"
                )
                
                analyze_btn = gr.Button("🔍 Analyze Emotion", variant="primary", size="lg")
                
                gr.HTML("<h3>💡 Try these examples:</h3>")
                gr.Examples(
                    examples=example_tweets,
                    inputs=[tweet_input],
                    label="Example Tweets"
                )
            
            with gr.Column(scale=2):
                result_output = gr.Markdown(label="📊 Analysis Results")
                confidence_plot = gr.BarPlot(
                    label="🎯 Confidence Scores",
                    x="Emotion",
                    y="Confidence",
                    color="Emotion",
                    title="Emotion Classification Confidence",
                    height=400
                )
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_tweet,
            inputs=[tweet_input],
            outputs=[result_output, confidence_plot]
        )
        
        tweet_input.submit(  # Allow Enter key to submit
            fn=analyze_tweet,
            inputs=[tweet_input],
            outputs=[result_output, confidence_plot]
        )
        
        # Footer
        gr.HTML("""
        <div style='text-align: center; margin-top: 20px; color: #666;'>
            <p>Built with ❤️ using Gradio and Hugging Face Transformers</p>
            <p>Upload your own emotion classification model to customize results!</p>
        </div>
        """)
    
    return interface

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=True,  # Creates a temporary public link
        server_name="0.0.0.0",  # Important for Hugging Face Spaces
        server_port=7860  # Default port for Hugging Face Spaces
    )