import PyPDF2
import chardet

def pdf_file_load(file_path):
    # Detect the encoding of the PDF file
    with open(file_path, 'rb') as pdf_file:
        result = chardet.detect(pdf_file.read())
        encoding = result['encoding']

    # Open the PDF file with the detected encoding
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    # If the detected encoding is not UTF-8, decode the text using the detected encoding
    if encoding and encoding.lower() != 'utf-8':
        text = text.encode(encoding).decode('utf-8', 'replace')

    return text


# split text into subsequences
def split_text(text, sub_len, step):
    sub_text = []
    for i in range(0, len(text), step):
        sub_text.append(text[i:i+sub_len])
    return sub_text


