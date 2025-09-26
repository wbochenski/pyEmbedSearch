from sentence_transformers import SentenceTransformer
import pandas as pd
import ast

def FindMostSimilarItem(df, query):
    cosine_similarities = []
    query_embedding = model.encode(query)

    for idx, row in df.iterrows():
        df_embedding = row["embedding"]
        
        similarity = model.similarity(df_embedding, query_embedding)

        cosine_similarities.append((df.loc[idx]["path"], float(similarity[0][0])))
    
    return sorted(cosine_similarities, key=lambda x: x[1], reverse=True)

CSV_PATH = 'data'

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

df = pd.read_csv(f"{CSV_PATH}.csv", sep=";")
df["embedding"] = df["embedding"].apply(ast.literal_eval)

while(1):
    query = input("Query: ")
    sorted_similarities = FindMostSimilarItem(df, query)
    print(sorted_similarities[0])
    print(sorted_similarities[1])
    print(sorted_similarities[2])
    print(sorted_similarities[3])
    print(sorted_similarities[4])
