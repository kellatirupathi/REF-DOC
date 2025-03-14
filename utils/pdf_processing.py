import pdfplumber

def extract_text_from_pdf(file_obj):
    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text