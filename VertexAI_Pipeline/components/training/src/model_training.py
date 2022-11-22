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
            
            
def hdbscan_model(training_data,job_id):
    '''
    This Function trains the HDBSCAN model
    
    Parameters:
        1. training_data(dataframe) : Dataframe where training data is stored
        2. job_id(string) : Vertex AI job id
    '''

    try:
        umap_embeddings = list(training_data['processed_embeddings'])
        cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                            metric='euclidean',                      
                            cluster_selection_method='eom',prediction_data=True).fit(umap_embeddings)

        with open('hdbscan_model', 'wb') as files:
            pickle.dump(cluster, files)
            
        upload_to_gcs(f'gs://vertexai-data-bucket1/{job_id}/models','hdbscan_model')

        return 1
  
    except Exception as e:

        return "Exception" + str(e)
    
    
def c_tf_idf(documents, m, ngram_range=(1, 1)):
    
    '''
    This Function performs count verctorisation and TFIDF 
    
    Parameters:
        1. documents(dataframe): Consist of a dataframe groupby object
        2. m(int): Lenght of training data
        3. ngram_range(tuple): NGram range
    '''
    
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_topic_sizes(df):
    
    '''
    This Function extracts and returns the topic sizes
    
    Parameters:
        1. df(dataframe): Dataframe containing training data
    '''
    
    topic_sizes = (df.groupby(['Topic'])
                     .content
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "content": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    
    return topic_sizes


def extract_top_words_per_topic(tf_idf, count, docs_per_topic, n=10):
    
    '''
    This Function extracts and returns the top n words per topic
    
    '''
    
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    # top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    top_n_words = {label: " ".join([words[j] for j in indices[i]][::-1]) for i, label in enumerate(labels)}
    return top_n_words


def topic_info_creation(training_data, job_id):
    '''
    This function creates topic info dataframe (cnsists of topic description and topic counts) and uplooads it to a GCS bucket
    
    Parameters:
        1. training_data(dataframe): Dataframe containing training data
        2. job_id(string) : Vertex AI job id
    '''

   # Loading Model
    with open('hdbscan_model' , 'rb') as f:
        cluster = pickle.load(f)
   # 
    training_data['content'] = training_data['content'].astype('str')
    docs_df = pd.DataFrame(training_data)
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'content': ' '.join})

    tf_idf, count = c_tf_idf(docs_per_topic.content.values, m=len(training_data))

    #Extraction Topic Count
    topics_count = extract_topic_sizes(training_data)

    #Extraction Topic description
    dictionary = extract_top_words_per_topic(tf_idf, count, docs_per_topic, n=10)

    # Outliers
    dictionary[-1] = 'Outliers'

    topics_desc = pd.DataFrame.from_dict(dictionary,orient='index',columns = ['topic_description']).reset_index()
    topics_desc.rename(columns = {'index':'Topic'},inplace = True)


    #Merging the two dataframes
    topic_info = pd.merge(topics_count,topics_desc,on = 'Topic',how = 'inner')

    topics_desc.to_csv(f'gs://vertexai-data-bucket1/{job_id}/training/topic_info.csv')
    
    topic_info_path = f'gs://vertexai-data-bucket1/{job_id}/training/topic_info.csv'
  
    return topic_info_path


def model_training_runner(job_id : str, topic_info_path : Output[Artifact]):
    
    '''
    This Function initiated the training for the HDBSCAN model and performs keyword extraction on the generated clusters
    
    Parameters:
        1. job_id(str) : Vertex AI job id
    
    '''
    
    try:
        training_data_path = f'gs://vertexai-data-bucket1/{job_id}/data_prep/data_for_training.csv'

        training_data = pd.read_csv(training_data_path, converters = {'processed_embeddings' : pd.eval})

        model_status = hdbscan_model(training_data,job_id)

        #print(model_status)

        #return model_status

        if model_status == 1 :

            trained_topic_info_path = topic_info_creation(training_data, job_id)

            topic_info_path.metadata = {"data_path" : trained_topic_info_path}

        else:

            topic_info_path.metadata = {"data_path" : "None : Model Training Failed, Error received :" + model_status}
    
    except Exception as e:

            return "Exception" + str(e)
        
    
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

 
