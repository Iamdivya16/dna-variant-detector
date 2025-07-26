import streamlit as st
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="variant_detector.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# One-hot mapping
mapping = {'A': [1, 0, 0, 0],
           'C': [0, 1, 0, 0],
           'G': [0, 0, 1, 0],
           'T': [0, 0, 0, 1]}

def one_hot_encode(seq):
    return np.array([mapping[base] for base in seq])

# Streamlit UI
st.title("🧬 DNA Variant Detector")
st.markdown("Paste a 50-letter DNA sequence to detect mutation-like patterns.")

user_input = st.text_input("🔠 Enter DNA Sequence (A/C/G/T only):", "")

if st.button("Predict"):
    if len(user_input) != 50 or any(c not in "ACGT" for c in user_input.upper()):
        st.error("❌ Please enter exactly 50 A/C/G/T letters.")
    else:
        seq = user_input.upper()
        encoded = one_hot_encode(seq)
        encoded = np.expand_dims(encoded, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], encoded)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

        st.success(f"✅ Prediction: {'Variant' if pred > 0.5 else 'Normal'} (Confidence: {pred:.3f})")
