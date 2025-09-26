from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import PyPDF2
from pathlib import Path
from typing import List, Tuple

def get_paths_to_files(path: Path) -> List[Path]:
    return [file for file in path.rglob('*') if file.is_file()]

def read_pdf(pdf_path: Path) -> str:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def split_text_into_chunks(text: str, max_chunk_length: int) -> List[str]:
    """ Split text into chunks while avoiding splitting in the middle of a sentence """
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
    return [[float(j) for j in i] for i in model.encode(chunks, show_progress_bar=False)]

def process_pdf(path: Path, model: SentenceTransformer, max_chunk_length: int = 256) -> List[Tuple[Path, List[float]]]:
    """Process a PDF, split into chunks, and generate embeddings"""
    content = read_pdf(path)
    chunks = split_text_into_chunks(content, max_chunk_length)
    embeddings = generate_embeddings(chunks, model)
    return [(path, embedding) for embedding in embeddings]

def main(data_folder: Path, model: SentenceTransformer):
    files = get_paths_to_files(data_folder)

    rows = []

    for path in tqdm(files):
        for path_embedding_tuple in process_pdf(path, model):
            rows.append(path_embedding_tuple)
    
    df = pd.DataFrame(rows, columns=['path', 'embedding'])
    df.to_csv(f'{data_folder}.csv', sep=";", index=False)

if __name__ == "__main__":
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    main(Path("data"), model)