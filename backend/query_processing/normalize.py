import re
from textblob import TextBlob

def normalize(query:str)->str:
    query = query.lower()
    query = re.sub(r"(\d)\.(\d)", r"\1<dot>\2", query)  # 6.5 → 6<dot>5
    query = re.sub(r"[^\w\s<dot>]", "", query)          # remove other punctuation
    query = re.sub(r"\s+", " ", query).strip()
    query = query.replace("<dot>", ".")  
    return query

# def correct_spelling(query):
#     return str(TextBlob(query).correct())

def preprocess_query(query):
    query = normalize(query)
    # query = correct_spelling(query)
    return query