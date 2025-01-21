import base64
import os
from io import BytesIO

import mammoth
from PyPDF2 import PdfReader, PdfWriter


def render_pdf(file_path: str, page_number: int) -> str:
    if not os.path.exists(file_path):
        raise ValueError(f"File path {file_path} does not exist")

    with open(file_path, "rb") as f:
        pdf_reader = PdfReader(f)

        if page_number < 1 or page_number > len(pdf_reader.pages):
            raise ValueError(
                f"Invalid page number. The PDF has {len(pdf_reader.pages)} pages."
            )

        page = pdf_reader.pages[page_number - 1]

        pdf_writer = PdfWriter()
        pdf_writer.add_page(page)

        output = BytesIO()
        pdf_writer.write(output)

        b64 = base64.b64encode(output.getvalue()).decode()

        href = f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='800px'></iframe>"

        return href


def render_docx(file_path: str, page_number: int) -> str:
    if not os.path.exists(file_path):
        raise ValueError(f"File path {file_path} does not exist")

    with open(file_path, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file)
        html = result.value
    # embedd html as iframe
    href = f"<iframe srcdoc='{html}' width='100%' height='800px'></iframe>"
    return href
