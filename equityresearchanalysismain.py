from flask import Flask, render_template, request, jsonify, send_file
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from fpdf import FPDF
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
summarization_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

file_path = "faiss_store.pkl"
docs_file_path = "docs_store_with_urls.pkl"


# Route to render the home page
@app.route('/')
def index():
    return render_template('index.html')


# Route to process URLs
@app.route('/process_urls', methods=['POST'])
def process_urls():
    urls = request.json.get('urls')
    if not urls:
        return jsonify({"error": "No URLs provided."}), 400

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }

        data = []
        url_mapping = []
        for url in urls:
            if url:
                try:
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text_content = soup.get_text()
                    data.append((text_content, url))
                except Exception as e:
                    return jsonify({"error": f"Error occurred with URL {url}: {e}"}), 400

        if not data:
            return jsonify({"error": "No data loaded from the URLs provided."}), 400

        documents = [Document(page_content=text) for text, _ in data]
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )

        docs = []
        for doc, url in data:
            split_docs = text_splitter.split_documents([Document(page_content=doc)])
            docs.extend(split_docs)
            url_mapping.extend([url] * len(split_docs))

        embeddings = []
        doc_contents = [doc.page_content for doc in docs]

        for doc in docs:
            embedding = embedding_model.encode(doc.page_content)
            embeddings.append(embedding)

        embeddings_np = np.array(embeddings).astype('float32')
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)

        with open(file_path, "wb") as f:
            pickle.dump(index, f)

        with open(docs_file_path, "wb") as f:
            pickle.dump((doc_contents, url_mapping), f, protocol=pickle.HIGHEST_PROTOCOL)

        return jsonify({"message": "FAISS vector store and documents saved successfully!"}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# Route to search for a query
@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided."}), 400

    try:
        with open(file_path, "rb") as f:
            index = pickle.load(f)
        with open(docs_file_path, "rb") as f:
            doc_contents, saved_urls = pickle.load(f)

        query_embedding = embedding_model.encode(query).reshape(1, -1)
        D, I = index.search(query_embedding, k=1)
        if len(I[0]) > 0:
            top_index = I[0][0]
            top_doc_content = doc_contents[top_index]
            top_doc_url = saved_urls[top_index]

            # Increase max_length and min_length for a longer summary
            summary = summarization_model(top_doc_content, max_length=300, min_length=50, do_sample=False)[0]['summary_text']
            combined_content = f"### EUREKA:\n{top_doc_content}\n\n### Summary:\n{summary}"

            # Create the PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, combined_content.encode('latin-1', 'replace').decode('latin-1'))

            pdf_file_path = "top_document_with_summary.pdf"
            pdf.output(pdf_file_path)

            # Send the document content and summary back in the response
            return jsonify({
                "message": "Search successful", 
                "summary": summary, 
                "doc_url": top_doc_url, 
                "pdf_file_path": pdf_file_path,
                "top_doc_content": top_doc_content  # Include the full document content in the response
            }), 200
        else:
            return jsonify({"error": "No relevant documents found."}), 404

    except Exception as e:
        return jsonify({"error": f"An error occurred during search: {str(e)}"}), 500


# Route to download the PDF
@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    pdf_file_path = request.args.get('pdf_file_path')
    if os.path.exists(pdf_file_path):
        return send_file(pdf_file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found."}), 404


if __name__ == '__main__':
    app.run(debug=True)
