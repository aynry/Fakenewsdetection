import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np
from streamlit_lottie import st_lottie

# Load the saved DistilBERT model and tokenizer
model = TFDistilBertForSequenceClassification.from_pretrained("bert_model")
tokenizer = DistilBertTokenizer.from_pretrained("bert_tokenizer")

# Define text preprocessing function
def preprocess_text(text, tokenizer, max_length=128):
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    return input_ids, attention_mask

# Streamlit app layout
def run():
    st.lottie("https://lottie.host/8285de03-31b0-4fce-98d7-e9e6fd886268/diocPQmJal.json", width=200, height=200)
    st.title("Fake News Detection with DistilBERT")
    st.text("Predict whether a news article is Real or Fake")
    
    user_input = st.text_area('Enter news content below:', placeholder='Input news content here...')
    st.text("")
    
    if st.button("Predict"):
        # Preprocess input and make prediction
        input_ids, attention_mask = preprocess_text(user_input, tokenizer)
        predictions = model.predict([input_ids, attention_mask])
        predicted_label = np.argmax(predictions.logits, axis=1).numpy()[0]
        
        # Interpret prediction result
        if predicted_label == 1:
            output = 'Real 👍'
        else:
            output = 'Fake 👎'
        
        st.success(f'The news article is likely: {output}')

if __name__ == "__main__":
    run()


st.markdown("**Created with ❤️ by Ayan**")