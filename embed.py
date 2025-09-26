from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import PyPDF2
from pathlib import Path

def get_paths_to_files(folder_path: str):
    folder = Path(folder_path)
    return [file for file in folder.rglob('*') if file.is_file()]

def read_pdf(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def split_text_into_chunks(text, max_chunk_length):
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

def generate_embeddings(text, model):
    embeddings = [model.encode(chunk).tolist() for chunk in text]
    return embeddings

def main(data_folder, model):
    df = pd.DataFrame({'path': [], 'embedding': []})
    files = get_paths_to_files(data_folder)

    for path in tqdm(files):
        content = read_pdf(path)
        content = split_text_into_chunks(content, 256)
        embeddings = generate_embeddings(content, model)

        new_rows = [(path, embedding) for embedding in embeddings]
        new_df = pd.DataFrame(new_rows, columns=['path', 'embedding'])

        df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv(f'{data_folder}.csv', sep=";", index=False)

if __name__ == "__main__":
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    main("data", model)