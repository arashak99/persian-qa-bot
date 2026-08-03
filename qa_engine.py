import fitz  # PyMuPDF
import easyocr
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)

#########################################################
# OCR
#########################################################

# Initialize once
reader = easyocr.Reader(
    ['fa', 'en'],
    gpu=torch.cuda.is_available()
)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a scanned PDF using
    PyMuPDF + EasyOCR.
    """

    doc = fitz.open(pdf_path)

    full_text = ""

    for page_num in range(len(doc)):

        page = doc.load_page(page_num)

        # Render page at higher resolution
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))

        img = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(
            pix.height,
            pix.width,
            pix.n
        )

        # Remove alpha channel if present
        if pix.n == 4:
            img = img[:, :, :3]

        results = reader.readtext(
            img,
            detail=0,
            paragraph=True
        )

        page_text = "\n".join(results)

        full_text += page_text + "\n"

    doc.close()

    return full_text.strip()


#########################################################
# Chunking
#########################################################

def chunk_text(
    text,
    tokenizer,
    max_length=384,
    stride=128,
):

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )

    input_ids = encoded["input_ids"]

    chunks = []

    for i in range(0, len(input_ids), stride):

        chunk = input_ids[i:i + max_length - 2]

        if len(chunk) < 10:
            continue

        chunks.append(chunk)

    return chunks


#########################################################
# QA
#########################################################

def answer_from_chunk(
    question,
    chunk_ids,
    tokenizer,
    model,
):

    question_ids = tokenizer.encode(
        question,
        add_special_tokens=False,
    )

    input_ids = (
        [tokenizer.cls_token_id]
        + question_ids
        + [tokenizer.sep_token_id]
        + chunk_ids
        + [tokenizer.sep_token_id]
    )

    input_ids = input_ids[:384]

    attention_mask = [1] * len(input_ids)

    while len(input_ids) < 384:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)

    input_tensor = torch.tensor([input_ids])
    mask_tensor = torch.tensor([attention_mask])

    with torch.no_grad():

        outputs = model(
            input_ids=input_tensor,
            attention_mask=mask_tensor,
        )

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    start = torch.argmax(start_logits).item()
    end = torch.argmax(end_logits).item()

    if start > end:
        start, end = end, start

    confidence = (
        start_logits[start].item()
        + end_logits[end].item()
    )

    answer = tokenizer.decode(
        input_ids[start:end + 1],
        skip_special_tokens=True,
    )

    return answer.strip(), confidence


#########################################################
# Main
#########################################################

def answer_question(
    question,
    pdf_path,
    tokenizer,
    model,
):

    print("Running OCR...")

    document = extract_text_from_pdf(pdf_path)

    print("OCR finished.")

    if len(document) == 0:
        return "OCR could not extract any text."

    chunks = chunk_text(document, tokenizer)

    best_answer = ""
    best_conf = float("-inf")

    for chunk in chunks:

        answer, conf = answer_from_chunk(
            question,
            chunk,
            tokenizer,
            model,
        )

        if conf > best_conf:
            best_conf = conf
            best_answer = answer

    if best_answer.strip() == "":
        return "پاسخی یافت نشد."

    return best_answer


#########################################################
# Test
#########################################################

if __name__ == "__main__":

    model_name = "mansoorhamidzadeh/parsbert-persian-QA"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name
    )

    model.eval()

    pdf_file = "فرم و تعهدنامه کامپیوتر کنکور1405 .pdf"

    question = "چند تعهد توی فرم بود؟"

    answer = answer_question(
        question,
        pdf_file,
        tokenizer,
        model,
    )

    print("\nQuestion:", question)
    print("Answer:", answer)