import pandas as pd
import logging
import json
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import nltk 
import string
import re
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from google.cloud import storage, bigquery
import umap.umap_ as umap
import hdbscan
import pickle
import csv
from sklearn.feature_extraction.text import CountVectorizer
from kfp.v2.dsl import Output, Artifact
from kfp.v2.components.executor import Executor
from sklearn.decomposition import PCA

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def clean_text(text,lemmatize_flag):
    
    '''
    Function takes plain text as input and returns clean text after removing stopwords, special characters and perform Lemmetization (optional)
    
    Parameters:
        1. text(string) : Plain text
        2. lemmatize_flag(bool) : Lemmatization Flag. "True" to perform lemmatization on the text, "False" to skip this step
        
    Returns:
        text(string) : Preprocessed Text
    '''

    stop_words = set(stopwords.words('english'))
    wn = nltk.WordNetLemmatizer() 

    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation

    text_rc = re.sub('[0-9]+', '', text_lc)

    if lemmatize_flag == 'True' :
        
        text_split = text_rc.split(" ")
        text = "".join([wn.lemmatize(word) for word in text_split if word not in stop_words])  # remove stopwords and stemming
    
        return text

    else:
      
        return text_rc


def upload_to_gcs(gcs_path: str, source_file_name: str) -> None:
    
    '''
    This Function uploads the specified fiels to the provide path of a GCS bucket
    
    Parameters:
        1. gcs_path(string) : GCS path where the file needs to be uploaded
        2. source_file_name(string) : name of the file that needs to be uploaded to the specified GCS path
    '''
      
        try:
            storage_client = storage.Client()
            path = gcs_path.split("//")[-1]
            bucket_name = path.split("/")[0]
            bucket = storage_client.get_bucket(bucket_name)
            if len(path.split("/")) > 1:
                folder_name = "/".join(path.split("/")[1:])
                blob = bucket.blob(folder_name + "/" + source_file_name)
            else:
                blob = bucket.blob(source_file_name)

            blob.upload_from_filename(source_file_name)

        except RuntimeError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.TracebackException(exc_type, exc_value, exc_tb)
            
            
def data_prep(sample_size : int ,lemmatize_flag : str, job_id : str , dim_reduction_algo : str ,data_path : Output[Artifact]):
    
    '''
    This Function prepares data for training by performing various preprocessing steps like 
    Data cleaning, Dimentionality reduction, and Conversion to embeddings.
    
    Parameters:
        1. sample_size(int) : Sample Size of the data to be preprocessed
        2. lemmatize_flag(str) : Lemmatization Flag. "True" to perform lemmatization on the text, "False" to skip this step
        3. job_id(str) : Vertex AI job id
        4. dim_reduction_algo(str) : Dimentionlaity Reduction algorithm from "PCA"/"UMAP". Default value "UMAP"
    
    '''
    
    try:
        data = pd.read_csv('gs://vertexai-data-bucket1/data.csv',delimiter=",", quoting=csv.QUOTE_NONE, encoding='utf-8',on_bad_lines='skip')
        data = data.sample(sample_size, replace=True)

        # Text Cleaning
        clean_data = []
        for text in list(data['content']):
            clean = clean_text(str(text),lemmatize_flag)
            clean_data.append(clean) 

        # Converting to Embeddings
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embeddings = list(model.encode(clean_data))

        with open('sentence_transformer_model', 'wb') as files:
            pickle.dump(model, files)

        upload_to_gcs(f'gs://vertexai-data-bucket1/{job_id}/models','sentence_transformer_model')

        if dim_reduction_algo == 'UMAP':

            # Dimentionality Reduction (UMAP)
            embedding_model = umap.UMAP(n_neighbors=15, 
                                    n_components=5, 
                                    metric='cosine')

        elif dim_reduction_algo == 'PCA':

            # Dimentionality Reduction (PCA)
            embedding_model = PCA(n_components=5)

        else:

            # Dimentionality Reduction (UMAP)---> Default
            embedding_model = umap.UMAP(n_neighbors=15, 
                                    n_components=5, 
                                    metric='cosine')



        processed_embeddings = embedding_model.fit_transform(embeddings)

        with open('embedding_model', 'wb') as files:
            pickle.dump(embedding_model, files)

        upload_to_gcs(f'gs://vertexai-data-bucket1/{job_id}/models','embedding_model')

        list_embeddings = [list(x) for x in embeddings]
        list_processed_embeddings = [list(x) for x in processed_embeddings]

        data['clean_data'] = clean_data
        data['embeddings'] = list(list_embeddings)
        data['processed_embeddings'] = list(list_processed_embeddings)

        data.to_csv(f'gs://vertexai-data-bucket1/{job_id}/data_prep/data_for_training.csv')

        preprocessed_data_path = f'gs://vertexai-data-bucket1/{job_id}/data_prep/data_for_training.csv'

        data_path.metadata = {"data_path" : preprocessed_data_path}
    
        #return data_path
    except Exception as e:

        return "Exception: " + str(e)


def executor_main():
    """
    The main executor function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor_input", type=str,required=True)
    parser.add_argument("--function_to_execute", type=str)
    args, _ = parser.parse_known_args()
    executor_input = json.loads(args.executor_input)
    function_to_execute = globals()[args.function_to_execute]
    print(function_to_execute)
    executor = Executor(
        executor_input=executor_input, function_to_execute=function_to_execute
    )
    executor.execute()


if __name__ == "__main__":
    executor_main()



