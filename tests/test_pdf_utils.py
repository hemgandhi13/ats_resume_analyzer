import fitz

from ats.pdf_utils import extract_text_from_pdf


def create_sample_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_extract_text_from_bytes():
    sample = "Hello, this is a test PDF."
    pdf_bytes = create_sample_pdf_bytes(sample)
    out = extract_text_from_pdf(pdf_bytes)
    assert sample.split()[0] in out
