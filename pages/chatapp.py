import os

import cv2
import numpy as np
import pdfplumber
import pytesseract
import streamlit as st
import torch
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from PIL import Image
from torchvision import models, transforms

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit configuration
st.set_page_config(page_title="PDF/Image Chatbot", page_icon=":scroll:")


# Load pre-trained model for image feature extraction
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Image transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Image enhancement using CLAHE and sharpening
def enhance_image(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Convert back to RGB
    enhanced_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

    # Apply sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_img, -1, kernel)

    return Image.fromarray(sharpened)


# Extract text from all pages of PDFs
def extract_pdf_text(pdf_docs):
    text = []
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_file:
            text.extend(
                [page.extract_text() for page in pdf_file.pages if page.extract_text()]
            )
    return "\n".join(text) if text else "No readable text found."


# Extract text from images using OCR
def extract_image_text(image_files):
    return "\n".join(
        [
            pytesseract.image_to_string(enhance_image(Image.open(img)))
            for img in image_files
        ]
    )


# Extract image features efficiently
def extract_image_features(image_files):
    features = []
    for img in image_files:
        image = enhance_image(Image.open(img)).convert("RGB")
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            features.append(
                model(image).squeeze().tolist()[:5]
            )  # Reduce memory footprint
    return features


# Create conversation chain
def get_conversational_chain():
    prompt = PromptTemplate(
        template="""
        Given the extracted image features and document context, provide a detailed explanation of the image.
        Context: {context}
        Image Features: {image_features}
        Question: {question}
        Answer:
        """,
        input_variables=["context", "image_features", "question"],
    )
    return LLMChain(llm=ChatOllama(model="qwen2.5"), prompt=prompt)


# Split text into chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    return splitter.split_text(text)


# Create FAISS vector store
def create_vector_store(text_chunks):
    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
    )
    vector_store.save_local("faiss_index")
    return vector_store


# Search FAISS store
def search_vector_store(query, vector_store):
    return "\n".join(
        [r.page_content for r in vector_store.similarity_search(query, k=7)]
    )


# Main function
def main():
    st.header("📚 Visual Data Interpreter System 🤖")
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input("Ask a Question about the uploaded files: ✍️")

    with st.sidebar:
        st.image(
            "E:/Multi-PDFs_ChatApp_AI-Agent-main/Multi-PDFs_ChatApp_AI-Agent-main/img/Robot.jpg"
        )
        st.title("📁 Upload Files")
        uploaded_files = st.file_uploader(
            "Upload PDF & Images",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if st.button("Process Files"):
            if not uploaded_files:
                st.error("Please upload at least one file.")
            else:
                with st.spinner("Processing..."):
                    pdfs = [
                        file for file in uploaded_files if file.name.endswith(".pdf")
                    ]
                    images = [
                        file
                        for file in uploaded_files
                        if file.name.endswith((".jpg", "jpeg", "png"))
                    ]

                    pdf_text = extract_pdf_text(pdfs)
                    image_text = extract_image_text(images)
                    image_features = extract_image_features(images)

                    combined_text = (
                        pdf_text + "\n\n[Image Extracted Text]\n" + image_text
                    )
                    text_chunks = chunk_text(combined_text)

                    st.session_state.vector_store = create_vector_store(text_chunks)
                    st.session_state.image_features = image_features
                    st.success("Processing complete!")

    if user_question:
        try:
            if st.session_state.vector_store is None:
                st.session_state.vector_store = FAISS.load_local(
                    "faiss_index",
                    OllamaEmbeddings(model="nomic-embed-text"),
                )

            context = search_vector_store(user_question, st.session_state.vector_store)
            image_features = st.session_state.get(
                "image_features", "No image features extracted."
            )

            chain = get_conversational_chain()
            response = chain.run(
                context=context, image_features=image_features, question=user_question
            )

            st.write("### 🤖 Visual Data Interpreter Response:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Please upload and process files before asking a question.")


if __name__ == "__main__":
    main()
