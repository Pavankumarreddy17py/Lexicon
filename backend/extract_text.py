import PyPDF2
# New import for handling DOCX files
import docx 
import os

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    doc = docx.Document(file_path)
    # Join all paragraph text with newlines
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text

def extract_text(file_path):
    """Extract text from various document file types."""
    file_path_lower = file_path.lower()
    file_extension = os.path.splitext(file_path_lower)[1]
    text = ""

    if file_extension == ".pdf":
        try:
            with open(file_path, "rb") as file:
                # Use PyPDF2's functionality, which is reasonably efficient page-by-page
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            return f"Error: Could not read PDF file. Ensure file is not corrupt. Details: {e}"
            
    elif file_extension == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                # Direct read for text files
                text = file.read()
        except Exception as e:
            return f"Error: Could not read TXT file. Details: {e}"
            
    elif file_extension == ".docx":
        try:
            text = extract_text_from_docx(file_path)
        except Exception as e:
            return f"Error: Could not read DOCX file. Details: {e}"

    # Add support for other common, simply readable formats (like code/data files)
    elif file_extension in [".csv", ".html", ".xml", ".json", ".md", ".py"]:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        except Exception as e:
            return f"Error: Could not read file of type {file_extension}. Details: {e}"

    else:
        return f"Error: Unsupported file type: {file_extension}. Currently supports PDF, TXT, and DOCX."
        
    return text.strip()