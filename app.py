from main import process_image_and_create_embeddings, process_pdf_and_create_embeddings, generate_response,check_chromadb_status
import streamlit as st
import google.generativeai as genai
import os
import tempfile

# Initialize session state variables if they don't exist
if 'compression_retriever' not in st.session_state:
    st.session_state.compression_retriever = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'chromadb_exists' not in st.session_state:
    st.session_state.chromadb_exists = False

# Configure API key
genai.configure(api_key='your_api_key')



def reset_state():
    """Reset all session state variables and clean up"""
    st.session_state.compression_retriever = None
    st.session_state.file_processed = False
    st.session_state.uploaded_file = None



def main():
    st.title("Document Question Answering System")

    # Check ChromaDB status
    if st.button("Check ChromaDB Status"):
        st.session_state.chromadb_exists = check_chromadb_status()
        if st.session_state.chromadb_exists:
            st.success("ChromaDB contains data. You can query without uploading a file.")
            st.session_state.compression_retriever = check_chromadb_status()
        else:
            st.warning("ChromaDB is empty. Please upload a file first.")

    # File upload section
    file_type = st.radio("Choose file type:", ["Image", "PDF"])
    uploaded_file = st.file_uploader(
        f"Upload your {file_type}",
        type=['pdf'] if file_type == "PDF" else ['png', 'jpg', 'jpeg']
    )

    # Process file button
    if uploaded_file and not st.session_state.file_processed:
        st.session_state.uploaded_file = uploaded_file
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                prompt = "Please carefully analyze the attached image and transcribe its text. The content is difficult to read, so apply multiple OCR passes and preprocessing techniques to achieve the most accurate result possible. Provide the data only"

                try:
                    if file_type == "PDF":
                        st.session_state.compression_retriever = process_pdf_and_create_embeddings(temp_file_path, prompt)
                    else:
                        st.session_state.compression_retriever = process_image_and_create_embeddings(temp_file_path, prompt)
                    st.session_state.file_processed = True
                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                finally:
                    os.unlink(temp_file_path)

    # Reset button
    if st.button("Reset"):
        reset_state()
        st.success("System reset successfully!")
        st.rerun()

    # Query section
    if st.session_state.compression_retriever:
        query = st.text_input("Enter your question:")
        if query:
            with st.spinner("Generating response..."):
                try:
                    response = generate_response(query, st.session_state.compression_retriever)
                    st.write("Response:", response)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()