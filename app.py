from flask import Flask, render_template, request, jsonify
import os
from werkzeug.exceptions import RequestEntityTooLarge
from backend.rag import RAGModel
# Update import name for the generalized function
from backend.extract_text import extract_text as extract_text_from_file

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Set configuration for max file size (50 MB) to handle large files
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 

# Handle the error specifically if file size is exceeded
@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(e):
    # This automatically handles files larger than MAX_CONTENT_LENGTH
    return jsonify({"error": "File size exceeds limit (50 MB)."}), 413

# Load Retrieval-Augmented Generation Model
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

    # Use the generalized extraction function
    extracted_text = extract_text_from_file(filename) 

    # Check for extraction error message (the custom function returns an "Error:" string on failure)
    if extracted_text.startswith("Error:"):
        # Cleanup the file that could not be processed
        try:
            os.remove(filename) 
        except OSError:
            pass # Ignore if file removal fails
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
    app.run(debug=True)