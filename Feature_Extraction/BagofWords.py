import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

class Bag_of_Words():
    
    def BOW_model(self,sentences):
        '''
        This Function performs BagofWords vectorization of the provided set of documents
        
        Parameters:
            1. sentences(list) : contens list of sentences for which vectorization needs to be performed
        
        '''
        try:
            vectorizer = CountVectorizer()
            count_matrix = vectorizer.fit_transform(sentences)
            count_array = count_matrix.toarray()
            df = pd.DataFrame(data=count_array,columns = vectorizer.get_feature_names_out())
            
            return df
        
        except Exception as e:

            return "Exception: " + str(e)
        
        
    def BOW_ngram_model(self,sentences,ngram_range):
        '''
        This Function performs BagofWords vectorization of the provided set of documents
        
        Parameters:
            1. sentences(list) : list of sentences for which vectorization needs to be performed
            2. ngram(tuple) : ngram_range =(1, 1) means only unigrams, ngram_range = (1, 2) means unigrams with bigrams ngram_range=(2, 2) means only bigrams.
        
        '''
        try:
            vectorizer = CountVectorizer(ngram_range = ngram_range) 
            count_matrix = vectorizer.fit_transform(sentences)
            count_array = count_matrix.toarray()
            df = pd.DataFrame(data=count_array,columns = vectorizer.get_feature_names_out())
            
            return df
        
        except Exception as e:

            return "Exception: " + str(e)
        
#Testing the BOW models
sentences = ["I work in Mumbai", "NLP is a niche skill to work on", "I will travel to London for work in a month", "I came to work here on nlp"]

model_obj = Bag_of_Words()


#BOW model
model_op = model_obj.BOW_model(sentences)
print(model_op)

#BOW ngram model
ngram_range = (1,2)
model_op = model_obj.BOW_ngram_model(sentences,ngram_range)
print(model_op)
        
        
        