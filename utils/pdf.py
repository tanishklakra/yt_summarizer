from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

def generate_pdf_summary(video_id, url, summary, output_dir="data/pdfs"):
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{video_id}_summary.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 40, "YouTube Video Summary")

    c.setFont("Helvetica", 12)
    c.drawString(40, height - 80, f"URL: {url}")

    text_obj = c.beginText(40, height - 120)
    text_obj.setFont("Helvetica", 12)
    text_obj.textLines("Summary:\n\n" + summary)
    c.drawText(text_obj)

    c.showPage()
    c.save()
    return pdf_path

def generate_detailed_pdf(video_id, explanation, output_dir="data/pdfs"):
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{video_id}_detailed.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 40, "Detailed Explanation")

    text_obj = c.beginText(40, height - 80)
    text_obj.setFont("Helvetica", 12)
    text_obj.textLines(explanation)
    c.drawText(text_obj)

    c.showPage()
    c.save()
    return pdf_path

def generate_breakdown_pdf(video_id, breakdown, output_dir="data/pdfs"):
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{video_id}_breakdown.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 40, "Time-Aligned Breakdown")

    text_obj = c.beginText(40, height - 80)
    text_obj.setFont("Helvetica", 12)
    text_obj.textLines(breakdown)
    c.drawText(text_obj)

    c.showPage()
    c.save()
    return pdf_path
