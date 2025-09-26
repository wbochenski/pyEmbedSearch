from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import PyPDF2
from pathlib import Path
from typing import List, Tuple

def get_paths_to_files(folder_path: Path) -> List[Path]:
    """Given a directory path, returns a list of all file paths (recursive search)."""
    return [path for path in folder_path.rglob('*') if path.is_file()]

def read_pdf(pdf_path: Path) -> str:
    """Reads a PDF file and extracts its text content."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def split_text_into_chunks(text: str, max_chunk_length: int) -> List[str]:
    """Splits a text into smaller chunks that are each less than or equal to `max_chunk_length` characters. 
    Tries to avoid splitting sentences in half."""
    sentences = text.split('. ')  
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) + 1 > max_chunk_length:
            chunks.append(chunk)
            chunk = sentence
        else:
            chunk = chunk + ". " + sentence if chunk else sentence
    if chunk:
        chunks.append(chunk)
    return chunks

def generate_embeddings(chunks: List[str], model: SentenceTransformer) -> List[List[float]]:
    """Generates embeddings for a list of text chunks using a SentenceTransformer model."""
    return [[float(value) for value in chunk_embedding] for chunk_embedding in model.encode(chunks, show_progress_bar=False)]

def process_pdf(pdf_path: Path, model: SentenceTransformer, max_chunk_length: int) -> List[Tuple[Path, List[float]]]:
    """Processes a PDF file: extracts the text, splits it into chunks, and generates embeddings."""
    pdf_content = read_pdf(pdf_path)
    chunks = split_text_into_chunks(pdf_content, max_chunk_length)  
    embeddings = generate_embeddings(chunks, model) 
    return [(pdf_path, embedding) for embedding in embeddings]  

def embed_every_pdf_in_path(folder_path: Path, model: SentenceTransformer, max_chunk_length: int) -> pd.DataFrame:
    """Processes all PDFs in a given folder, generates embeddings, and returns the results as a pandas DataFrame."""
    paths = get_paths_to_files(folder_path) 
    rows = []  

    for path in tqdm(paths):  
        for path_embedding_tuple in process_pdf(path, model, max_chunk_length):  
            rows.append(path_embedding_tuple)  
    
    return pd.DataFrame(rows, columns=['path', 'embedding'])  

if __name__ == "__main__":
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  
    df = embed_every_pdf_in_path(Path("data"), model, 256)  
    df.to_csv('data.csv', sep=";", index=False)
