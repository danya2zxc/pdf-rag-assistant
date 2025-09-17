from PyPDF2 import PdfReader


def pdf_reader(file):
    text = PdfReader(file)

    return text
