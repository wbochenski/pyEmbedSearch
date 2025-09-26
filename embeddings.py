from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd

def GetPathsToFilesInFolder(folder_path: str):
    import os
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    return file_paths
def ReadPDF(pdf_path: str) -> str:
    import PyPDF2
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

FOLDER_PATH = 'data'
MAX_LENGTH = 256

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
df = pd.DataFrame({'path': [], 'embedding': []})
files = GetPathsToFilesInFolder(FOLDER_PATH)

for path in tqdm(files):
    content = ReadPDF(path)
    split_content = [content[i:i + MAX_LENGTH] for i in range(0, len(content), int(MAX_LENGTH/2))]

    embeddings = [model.encode(chunk).tolist() for chunk in split_content]

    new_rows = [(path, embedding) for embedding in embeddings]
    new_df = pd.DataFrame(new_rows, columns=['path', 'embedding'])

    df = pd.concat([df, new_df], ignore_index=True)

df.to_csv(f'{FOLDER_PATH}.csv', sep=";", index=False)