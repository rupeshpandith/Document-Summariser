import gradio as gr
import fitz  # PyMuPDF
from transformers import pipeline

# Use google/pegasus-cnn_dailymail for high-quality, fast summarization
summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with fitz.open(pdf_file.name) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        return f"Error extracting text from PDF: {e}"
    return text

def summarize_document(file):
    try:
        raw_text = extract_text_from_pdf(file)
        if raw_text.startswith("Error extracting text"):
            return raw_text
        if not raw_text.strip():
            return "No text found in the PDF."
        max_input_length = 1024
        truncated_text = raw_text[:3000]
        summary = summarizer(truncated_text, max_length=150, min_length=40, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error during summarization: {e}"

# Custom CSS for copy button
custom_css = '''
#summary-container {
    position: relative;
    margin-bottom: 1em;
}
'''

def reset_interface():
    return None, ""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 📄 PDF Document Summarizer\nUpload a PDF file. The app extracts the text and returns a short, clear summary.")
    with gr.Row():
        file_input = gr.File(label="Upload your PDF")
        upload_again_btn = gr.Button("Upload again", elem_id="upload-again-btn")
    with gr.Row():
        with gr.Column(elem_id="summary-container"):
            summary_output = gr.Textbox(label="Summary", lines=8, interactive=False, elem_id="summary-box")
    file_input.change(summarize_document, inputs=file_input, outputs=summary_output)
    upload_again_btn.click(fn=reset_interface, inputs=None, outputs=[file_input, summary_output])

demo.launch()
