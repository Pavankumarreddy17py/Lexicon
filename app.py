from flask import Flask, render_template, request, jsonify
import os
from werkzeug.exceptions import RequestEntityTooLarge
from backend.rag import RAGModel
from backend.extract_text import extract_text as extract_text_from_file

# For local running, keep the default template/static folders
app = Flask(__name__)

# Ensure the local upload folder exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 

@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(e):
    return jsonify({"error": "File size exceeds limit (50 MB)."}), 413

# Initialize the RAG model
print("Initializing RAG Model...")
rag_model = RAGModel()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filename)

    extracted_text = extract_text_from_file(filename) 

    if extracted_text.startswith("Error:"):
        try:
            os.remove(filename) 
        except OSError:
            pass
        return jsonify({"error": extracted_text}), 500

    rag_model.add_document(filename, extracted_text)
    return jsonify({"message": "File uploaded successfully"}), 200

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = rag_model.answer_question(question)
    return jsonify({"answer": answer}), 200

if __name__ == "__main__":
    print("Starting Flask local server...")
    # Using 127.0.0.1 for local development
    app.run(host='127.0.0.1', port=5000, debug=True)