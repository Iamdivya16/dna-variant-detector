import streamlit as st
import numpy as np
import tflite_runtime.interpreter as tflite
from collections import Counter

# Load TFLite model
interpreter = tflite.Interpreter(model_path="variant_detector.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# One-hot mapping
mapping = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1]
}


def one_hot_encode(seq):
    return np.array([mapping[base] for base in seq])


# Streamlit UI
st.title("🧬 DNA Variant Detector")
st.markdown(
    "Paste a 50-letter DNA sequence to detect mutation-like patterns."
)

user_input = st.text_input(
    "🔠 Enter DNA Sequence (A/C/G/T only):",
    ""
)

if st.button("Predict"):

    seq = user_input.strip().upper()

    # Validation
    if len(seq) != 50:
        st.error("❌ Please enter exactly 50 nucleotides.")
    elif any(base not in "ACGT" for base in seq):
        st.error("❌ Sequence can contain only A, C, G, and T.")
    else:
        # GC Content
        gc_content = (
            seq.count("G") + seq.count("C")
        ) / len(seq) * 100

        # Nucleotide counts
        counts = Counter(seq)

        # Model prediction
        encoded = one_hot_encode(seq)
        encoded = np.expand_dims(
            encoded,
            axis=0
        ).astype(np.float32)

        interpreter.set_tensor(
            input_details[0]['index'],
            encoded
        )

        interpreter.invoke()

        pred = float(
            interpreter.get_tensor(
                output_details[0]['index']
            )[0][0]
        )

        # Prediction result
        st.subheader("Prediction Result")

        if pred > 0.5:
            st.success(
                f"✅ Variant Detected "
                f"({pred * 100:.2f}% confidence)"
            )
        else:
            st.info(
                f"✅ Normal Sequence "
                f"({(1 - pred) * 100:.2f}% confidence)"
            )

        # GC Content
        st.metric(
            label="GC Content",
            value=f"{gc_content:.2f}%"
        )

        # Nucleotide distribution
        st.subheader("Nucleotide Distribution")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("A", counts.get("A", 0))
        col2.metric("C", counts.get("C", 0))
        col3.metric("G", counts.get("G", 0))
        col4.metric("T", counts.get("T", 0))
