from sentence_transformers import SentenceTransformer
import pandas as pd
import ast

def load_data(csv_path):
    df = pd.read_csv(f"{csv_path}.csv", sep=";")
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    return df

def get_list_of_paths_sorted_by_similarity(model, df, query):
    cosine_similarities = []
    query_embedding = model.encode(query)

    for idx, row in df.iterrows():
        df_embedding = row["embedding"]
        similarity = model.similarity(df_embedding, query_embedding)
        cosine_similarities.append((df.loc[idx]["path"], float(similarity[0][0])))
    
    return sorted(cosine_similarities, key=lambda x: x[1], reverse=True)

def main(csv_path, model, query):
    df = load_data(csv_path)
    sorted_similarities = get_list_of_paths_sorted_by_similarity(model, df, query)
    print(sorted_similarities[0])

if __name__ == "__main__":
    query = input("Search: ")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    main("data", model, query)