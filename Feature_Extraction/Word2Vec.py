import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec

nltk.download('punkt')

class Word2Vec():
        
    def CBOW_model(self,training_data_path, min_count, window):
        
        '''
        This function is used to train a continuous bag of words (CBOW) Word2Vec model
        
        Parameters:
            1. training_data_path(string): path of the .txt file used for training the Word2Vec model
            2. min_count(int): (Gensim Word2vec model parameter) Ignores all words with total frequency lower than this.
            3. window(int): (Gensim Word2vec model parameter) Maximum distance between the current and predicted word within a sentence.
        '''
        try:
            # Reads txt file
            sample = open(training_data_path)
            s = sample.read()

            # Replaces escape character with space
            f = s.replace("\n", " ")

            data = []
            # iterate through each sentence in the file
            for i in sent_tokenize(f):
                temp = []
                # tokenize the sentence into words
                for j in word_tokenize(i):
                    temp.append(j.lower())

                data.append(temp)

            # Create CBOW model
            model = gensim.models.Word2Vec(data, min_count = min_count, window = window)
            
            return model
        
        
        except Exception as e:

            return "Exception: " + str(e)
        
    def Skip_Gram_model(self,training_data_path, min_count, window):
        
        '''
        This function is used to train a SkipGram Word2Vec model
        
        Parameters:
            1. training_data_path(string): path of the .txt file used for training the Word2Vec model
            2. min_count(int): (Gensim Word2vec model parameter) Ignores all words with total frequency lower than this.
            3. window(int): (Gensim Word2vec model parameter) Maximum distance between the current and predicted word within a sentence.
        '''
        try:
            # Reads txt file
            sample = open(training_data_path)
            s = sample.read()

            # Replaces escape character with space
            f = s.replace("\n", " ")

            data = []
            # iterate through each sentence in the file
            for i in sent_tokenize(f):
                temp = []
                # tokenize the sentence into words
                for j in word_tokenize(i):
                    temp.append(j.lower())

                data.append(temp)

            # Create CBOW model
            model = gensim.models.Word2Vec(data, min_count = min_count, window = window, sg = 1)
            
            return model
        
        except Exception as e:

            return "Exception: " + str(e)
        
#Testing the Word2Vec models
training_data_path = "word2vec_training_data_aliceinwonderland.txt"
min_count = 1
window = 5

Word2Vec_obj = Word2Vec()

#CBOW
model = Word2Vec_obj.CBOW_model(training_data_path, min_count, window)
print("Cosine similarity - CBOW : ", model.wv.similarity('alice', 'wonderland'))
print("Cosine similarity - CBOW : ", model.wv.similarity('alice', 'machines'))

#SkipGram
model = Word2Vec_obj.Skip_Gram_model(training_data_path, min_count, window)
print("Cosine similarity - SkipGram : ", model.wv.similarity('alice', 'wonderland'))
print("Cosine similarity - SkipGram : ", model.wv.similarity('alice', 'machines'))
