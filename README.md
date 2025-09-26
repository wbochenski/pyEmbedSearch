# pyEmbeddedSearch

This project uses natural language processing (NLP) techniques to build a search engine that can find the most similar content to a user query based on PDF documents. The core of this system is based on Sentence Transformers to generate embeddings from document contents, which are then compared to query embeddings to retrieve the most relevant results.

# 📋 Overview

The AI-Search project performs the following tasks:

- Preprocessing PDFs: It reads a folder of PDF files, extracts text from each document, and splits the content into chunks.

- Generating Embeddings: For each text chunk, it computes sentence embeddings using the Sentence Transformer model. 

- Storing Data: The paths of the PDFs along with their corresponding embeddings are stored in a CSV file.

- Querying: Once the embeddings are stored, you can query the system by entering a text input. The system computes the query's embedding and finds the most similar documents based on cosine similarity.

# 🗂️ Project Structure ️

- embeddings.py: This script processes all PDF files in a folder, generates embeddings for their content, and stores them in a CSV file.

- query.py: This script allows you to input a query and retrieves the most similar documents based on their embeddings stored in the CSV.

# ⚙️ Requirements ️

- Python 3.x

- sentence-transformers library

- tqdm for progress bars

- pandas for handling CSV data

- PyPDF2 for reading PDF files

You can install the dependencies with:

```bash
pip install sentence-transformers tqdm pandas PyPDF2
```

# 🚀 Usage 

## 📄 Generate Embeddings 

Run the embeddings.py script to process the PDF files from the folder data/:

```bash
python embeddings.py
```

This script:

- Reads all PDFs from the data/ folder.

- Extracts text from each PDF.

- Generates sentence embeddings for chunks of text.

- Saves the file paths and their corresponding embeddings into a CSV file data.csv.

## 🔎 Query the Database 

Once the embeddings are generated and saved, you can search for the most similar documents by running the query.py script:

```bash
python query.py
```

This will prompt you to enter a query, and it will display the top 5 most similar document paths based on cosine similarity of the embeddings.
