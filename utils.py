import os
from pathlib import Path
from typing import Iterable

import docx2txt
import fitz
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file: FileStorage, upload_folder: str) -> str:
    if not file or not file.filename:
        raise ValueError("No file was uploaded.")
    if not allowed_file(file.filename):
        raise ValueError("Unsupported file type. Upload PDF, DOCX, or TXT.")

    os.makedirs(upload_folder, exist_ok=True)
    filename = secure_filename(file.filename)
    path = os.path.join(upload_folder, filename)
    file.save(path)
    return path


def extract_text_from_pdf(file_path: str) -> str:
    text_parts = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def extract_text_from_docx(file_path: str) -> str:
    return docx2txt.process(file_path) or ""


def extract_text_from_txt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


def extract_text(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()
    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    if extension == ".docx":
        return extract_text_from_docx(file_path)
    if extension == ".txt":
        return extract_text_from_txt(file_path)
    raise ValueError("Unsupported file type.")


def require_text(value: str, field_name: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"{field_name} is required.")
    return value


def join_nonempty(parts: Iterable[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())
