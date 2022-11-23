from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from math import exp

class Text_similarity():

    def jaccard_similarity(self, sentences):
        '''
        The Function computes the jackard similarity between the sentences
        '''
        try:
            
            sentences = [sent.lower().split(" ") for sent in sentences]
            intersection = len(set.intersection(*[set(i) for i in sentences]))
            union = len(set.union(*[set(i) for i in sentences]))

            return intersection/float(union)
        
        except Exception as e:

            return "Exception: " + str(e)
        
    def eucledian_distance_normalized(self, arr):
        '''
        The Function computes eucledian distance between the elements of the vectorized text array
        '''
        try:
            euclidean_arr = euclidean_distances(arr)
            nrm_euclidean_distances = [[1/exp(i) for i in euclidean_arr[x]] for x in range(len(euclidean_arr))]
            
            return nrm_euclidean_distances
            
        except Exception as e:

            return "Exception: " + str(e)
        
    def cosine_similarity(self, arr):
        '''
        The Function computes cosine similarity between the elements of the vectorized text array
        '''
        try:
            cos_similarity = cosine_similarity(arr)
            
            return cos_similarity
            
        except Exception as e:

            return "Exception: " + str(e)
        
#Testing
Text_similarity_obj = Text_similarity()

#Jaccard Similarity
sentences = ["The bottle is empty","The bottle is not empty"]

print(Text_similarity_obj.jaccard_similarity(sentences))

#Eucledian distance & Cosine Similarity

sentences = [
#Crypto
'Investors unfazed by correction as crypto funds see $154 million inflows',
'Bitcoin, Ethereum prices continue descent, but crypto funds see inflows',
 
#Inflation
'The surge in euro area inflation during the pandemic: transitory but with upside risks',
"Inflation: why it's temporary and raising interest rates will do more harm than good",
 
#common
'Will Cryptocurrency Protect Against Inflation?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
arr = X.toarray()

#Eucledian Distance
print(Text_similarity_obj.eucledian_distance_normalized(arr))

#Cosine Similarity
print(Text_similarity_obj.cosine_similarity(arr))


