import pickle
import pandas as pd



def extract_features(df, query):
    df = df.copy()
    df["title_len"] = df["title"].apply(lambda x: len(str(x).split()))
    df["desc_len"] = df["description"].apply(lambda x: len(str(x).split()))
    df["query_category_match"] = df["main_category"].apply(
        lambda x: 1 if str(x).lower() in query.lower() else 0
    )
    return df[[
        "semantic_score", "average_rating", "rating_number",
        "price", "title_len", "desc_len", "query_category_match"
    ]]
