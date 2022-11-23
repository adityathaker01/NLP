from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def TfIdf_vecorizer(sentences):
    '''
    This Function performs TFidf vectorization of the provided set of documents 
    
    '''
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(sentences)
    count_array = matrix.toarray()
    df = pd.DataFrame(data=count_array,columns = vectorizer.get_feature_names_out())
    
    return df


sentences = ["I work in Mumbai", "NLP is a niche skill to work on", "I will travel to London for work in a month", "I came to work here on nlp"]

print(TfIdf_vecorizer(sentences))