import pdfplumber

def extract_text(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text.strip()

# Change the path to a real Persian PDF you have
text = extract_text("فرم و تعهدنامه کامپیوتر کنکور1405 .pdf")
print(text[500:])   # first 500 characters
