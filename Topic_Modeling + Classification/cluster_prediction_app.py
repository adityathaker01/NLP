from flask import Flask, jsonify, request
import pickle
import pandas as pd
import hdbscan

app = Flask(__name__)

def cluster_prediction(text):
    '''
    This Function compiles the result of the cluster prediction
    '''
    
    #model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    with open('transformer_model' , 'rb') as f:
        
        model = pickle.load(f)
    
    embeddings = list(model.encode([text]))
    
    with open('embeddings_model' , 'rb') as f:
        
        
        embeddings_model = pickle.load(f)

    # Loading embeddings model
    processed_embeddings = embeddings_model.transform(embeddings)

    
    # Loading HDBSCAN model
    with open('hdbscan_model' , 'rb') as f:
        
        cluster = pickle.load(f)

    topics = set(list(cluster.labels_))

    test_labels, strengths = hdbscan.approximate_predict(cluster, processed_embeddings)
    topic = test_labels[0]
    df = pd.read_csv('topic_info.csv')
    df = df[df['Topic']==topic]
    description = list(df['topic_description'])[0]
    result = {'topic' : int(topic), 'description': description}

    return result

# routes
@app.route('/', methods=['POST'])
def predict():
    """Function is used for prediction"""
    # get data 
    data = request.get_json()['text'] 

    # predictions
    result = cluster_prediction(data)
    
    # send back to browser
    output = {'results': result}
    
    return output


if __name__ == '__main__':
    app.run(port=5000, debug=True)


# !python3 cluster_prediction_app.py
# !curl http://127.0.0.1:5000/ -d "{\"text\": \" i have not go fjisd dhsl \"} " -H 'Content-Type: application/json'
