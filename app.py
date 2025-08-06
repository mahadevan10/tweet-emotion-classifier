import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import numpy as np

# Load your emotion classification model
MODEL_NAME = "mahadevan10/tweet-emotion-classifier/sentiment_model"

def load_model():
    """Load the emotion classification model"""
    try:
        print(f"Loading model: {MODEL_NAME}")
        # Try loading your specific model
        classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            return_all_scores=True
        )
        print("Model loaded successfully!")
        return classifier
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        print("Trying without return_all_scores...")
        try:
            # Try without return_all_scores
            classifier = pipeline(
                "text-classification",
                model=MODEL_NAME
            )
            print("Model loaded without return_all_scores!")
            return classifier
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            print("Using fallback model...")
            # Fallback to a working model for demo
            classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            print("Fallback model loaded!")
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
        print(f"Analyzing tweet: {tweet_text[:50]}...")
        
        if not tweet_text or not tweet_text.strip():
            return "Please enter a tweet to analyze! 🐦", {}
        
        # Preprocess the tweet
        processed_tweet = preprocess_tweet(tweet_text)
        print(f"Processed tweet: {processed_tweet[:50]}...")
        
        if len(processed_tweet) < 3:
            return "Tweet too short to analyze. Please enter a longer tweet. 📝", {}
        
        # Get predictions
        print("Making prediction...")
        results = emotion_classifier(processed_tweet)
        print(f"Raw results: {results}")
        
        # Handle different result formats
        if isinstance(results, list) and len(results) > 0:
            # Check if it's nested list or direct list
            emotions = results[0] if isinstance(results[0], list) else results
            print(f"Emotions extracted: {emotions}")
            
            # Create a formatted output
            output_text = "🎭 **Emotion Analysis Results:**\n\n"
            output_text += f"**Tweet:** {tweet_text[:100]}{'...' if len(tweet_text) > 100 else ''}\n\n"
            
            # Handle single prediction vs multiple scores
            if isinstance(emotions, dict):
                # Single prediction format
                emotion_name = emotions.get('label', 'Unknown').title()
                confidence = emotions.get('score', 0)
                output_text += f"🏆 **Primary Emotion: {emotion_name}** ({confidence:.2%})\n"
                confidence_dict = {emotion_name: confidence}
            elif isinstance(emotions, list):
                # Multiple predictions format
                # Sort emotions by confidence
                sorted_emotions = sorted(emotions, key=lambda x: x.get('score', 0), reverse=True)
                
                # Create confidence dictionary for the plot
                confidence_dict = {}
                
                for i, emotion in enumerate(sorted_emotions):
                    emotion_name = emotion.get('label', 'Unknown').title()
                    confidence = emotion.get('score', 0)
                    confidence_dict[emotion_name] = confidence
                    
                    if i == 0:  # Highest confidence
                        output_text += f"🏆 **Primary Emotion: {emotion_name}** ({confidence:.2%})\n"
                    else:
                        output_text += f"   {emotion_name}: {confidence:.2%}\n"
            else:
                return f"Unexpected result format: {type(emotions)}", {}
            
            print(f"Final output: {output_text}")
            print(f"Confidence dict: {confidence_dict}")
            return output_text, confidence_dict
        else:
            print(f"No results or empty results: {results}")
            return "No emotion detected. Please try a different tweet. ⚠️", {}
            
    except Exception as e:
        print(f"Error in classify_emotion: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error occurred: {str(e)} ❌\n\nPlease check the logs for more details.", {}

def analyze_tweet(tweet_text):
    """
    Main function called by Gradio interface
    """
    result_text, confidence_scores = classify_emotion(tweet_text)
    
    # Create a detailed text output for all emotions
    if confidence_scores:
        emotions_text = "Detailed Emotion Breakdown:\n\n"
        sorted_emotions = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, score in sorted_emotions:
            percentage = score * 100
            bar = "█" * int(percentage / 5)  # Visual bar
            emotions_text += f"{emotion:12} {percentage:6.2f}% {bar}\n"
    else:
        emotions_text = "No emotions detected."
    
    return result_text, emotions_text

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
        gr.HTML("<h1 style='text-align: center; color: #1DA1F2;'>🐦 Mahadevan's Tweet Emotion Classifier 🎭</h1>")
        gr.HTML("<p style='text-align: center; font-size: 18px;'>Analyze emotions in tweets using my custom AI model! Enter any tweet below and discover its emotional tone.</p>")
        
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
                
                # Simple text output for emotions (always works)
                emotions_output = gr.Textbox(
                    label="🎭 All Emotions with Scores",
                    lines=8,
                    max_lines=10,
                    interactive=False
                )
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_tweet,
            inputs=[tweet_input],
            outputs=[result_output, emotions_output]
        )
        
        tweet_input.submit(  # Allow Enter key to submit
            fn=analyze_tweet,
            inputs=[tweet_input],
            outputs=[result_output, emotions_output]
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