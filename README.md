# 📄 Arabic RAG - Document Question Answering System

## 🚀 Overview
This project is a **Retrieval-Augmented Generation (RAG)** system for answering questions based on uploaded documents. It supports:

📷 **Images** (PNG, JPG, JPEG)  
📄 **PDF Documents**

## 🛠️ Installation

### 🔽 Clone the repository:
```bash
git clone https://github.com/your-repo/omar-yyoussef-arabic_rag.git
cd omar-yyoussef-arabic_rag
```

### 🏗️ Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 📦 Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚠️ Poppler Installation (Required for PDF Processing)
Since **pdf2image** is used, **Poppler** must be installed.

### 🖥️ Windows
1. Download the latest **Poppler** release from:  
   🔗 [Poppler for Windows](https://poppler.freedesktop.org/)
2. Extract the downloaded archive.
3. Set the `poppler_path` in `main.py`:
   ```python
   poppler_path = r"C:\path\to\poppler\Library\bin"
   ```

### 🐧 Linux
```bash
sudo apt update && sudo apt install poppler-utils
```

### 🍏 macOS
```bash
brew install poppler
```

## 🎮 Usage
Run the **Streamlit** app:
```bash
streamlit run app.py
```

### 📤 Upload a file:
✅ Choose between **PDF or Image**.  
✅ Upload your file using the **file uploader**.

### 🔍 Process the document:
🚀 Click the **"Process Document"** button to extract text and create embeddings.

### ❓ Ask questions:
📝 Enter your question in the **text input field**.  
💡 The system will generate a response based on the extracted text.

📂 **No upload required if data exists in ChromaDB**:
🔹 If ChromaDB already has stored data, you can immediately start asking questions **after clicking "Check ChromaDB Status"**.

### 🔄 Reset the system:
🔃 Use the **"Reset"** button to clear the session and start over.

---

## 🧠 How It Works

### 🔡 OCR (Optical Character Recognition):
📷 For **images**, the system uses **Google's Gemini AI** to extract text.  
📄 For **PDFs**, the system converts each page to an **image** and then performs **OCR**.

### 🔀 Text Chunking:
📌 The extracted text is split into **smaller chunks** for efficient processing.

### 🏗️ Embeddings:
🧩 Text chunks are converted into **embeddings** using **HuggingFace's AraBert model**.

### 📦 Vector Storage:
📂 Embeddings are stored in **ChromaDB** for fast retrieval.

### ❓ Question Answering:
🔎 When a **question** is asked, the system **retrieves relevant chunks** using a **compression retriever** and generates a **response** using **Google's Gemini AI**.

---

## 📂 Directory Structure
```
omar-yyoussef-arabic_rag/
├── app.py               # Streamlit application
├── config.py            # Configuration file for API keys, model paths, etc.
├── main.py              # Core logic for processing images, PDFs, and embeddings
├── requirements.txt     # Dependencies
├── models/              # Folder for model-related files
│   ├── embedding_model.py   # Embedding model (e.g., AraBert)
│   ├── reranker_model.py    # Reranker model
│   └── __pycache__/      # Cached files for compiled models
└── utils/               # Helper functions
    ├── ocr_utils.py     # OCR-related utility functions
    ├── retriever.py     # Functions related to document retrieval
    ├── text_processing.py  # Text chunking and processing utilities
    ├── vector_store.py  # Functions for working with ChromaDB and vector storage
    └── __pycache__/      # Cached utility files
```

---

## 📝 Requirements
✔️ **Python 3.8+**  
✔️ **Poppler** (for PDF processing)  
✔️ **Google API Key** (for Gemini AI)

---

## 🤖 Models Used
🔹 **Embedding Model:** [Omartificial-Intelligence-Space--GATE-AraBert-v1](https://huggingface.co/Omartificial-Intelligence-Space/GATE-AraBert-v1)  
🔹 **Reranker Model:** [Omartificial-Intelligence-Space--ARA-Reranker-V1](https://huggingface.co/Omartificial-Intelligence-Space/ARA-Reranker-V1)  
🔹 **Generative Model:** [Google's Gemini 2.0 Flash](https://ai.google.dev/)  

---

## 📜 License
This project is licensed under the **MIT License**. Feel free to **use, modify, and distribute** it as per the license terms.

---

## 🙏 Acknowledgments
💙 **Google** for the **Gemini AI**.  
💡 **LangChain** for the **RAG framework**.  
🤗 **HuggingFace** for the **embedding and reranker models**.  

🎯 **Happy Coding! 🚀**

