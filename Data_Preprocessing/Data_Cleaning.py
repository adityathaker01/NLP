import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from cleantext import clean

nltk.download("stopwords")

class Text_Cleaning():
    
    def __init__(self, text):
        """Initializes text cleaning"""
    
        self.text = text
    
    def remove_stopwords(self,text):
        '''
        Function returns text after removing stopwords from plain text
        
        Parameters:
            1. text(string) : Plain text
            
        Returns:
            text(string) : Text after removing stopwords
        '''
        
        stop_words = set(stopwords.words('english'))
        text_split = text.split(" ")
        text = " ".join([word.lower() for word in text_split if word not in string.punctuation and word not in stop_words])
        
        return text
    
    def lemmatize_text(self,text):
        '''
        Function returns text after lemmatizing words from plain text
        
        Parameters:
            1. text(string) : Plain text
            
        Returns:
            text(string) : Text after lemmatization
        '''
        
        wn = nltk.WordNetLemmatizer()
        text_split = text.split(" ")
        text = " ".join([wn.lemmatize(word) for word in text_split if word not in stop_words])  # remove stopwords and lemmetizing
        
        return text
        
        
    
    def clean_text(self,lemmatize_flag):
        '''
        Function takes plain text as input and returns clean text after removing stopwords, special characters and perform Lemmetization (optional)

        Parameters:
            1. text(string) : Plain text
            2. lemmatize_flag(bool) : Lemmatization Flag. "True" to perform lemmatization on the text, "False" to skip this step

        Returns:
            text(string) : Preprocessed Text
        '''
        try:

            #Using clean-text library to clean the text data
            text = clean(self.text,
                    #fix_unicode=True,              # fix various unicode errors
                    to_ascii=True,                  # transliterate to closest ASCII representation
                    lower=True,                     # lowercase text
                    no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
                    no_urls=True,                   # replace all URLs with a special token
                    no_emails=True,                 # replace all email addresses with a special token
                    no_phone_numbers=True,          # replace all phone numbers with a special token
                    no_numbers=False,               # replace all numbers with a special token
                    no_digits=True,                 # replace all digits with a special token
                    no_currency_symbols=True,       # replace all currency symbols with a special token
                    no_punct=True,                  # remove punctuations
                    replace_with_punct="",          
                    replace_with_url="",
                    replace_with_email="",
                    replace_with_phone_number="",
                    replace_with_number="",
                    replace_with_digit="",
                    replace_with_currency_symbol="<CUR>",
                    lang="en"                      
                        )

            # Removing Stopwords
            text = self.remove_stopwords(text)

            # Lemmetizing Text
            if lemmatize_flag == True :

                text = self.lemmatize_text(text)

                return text

            else:

                return text

        except Exception as e:

            return "Exception: " + str(e)

#Testing the code for Data Cleaning
text = "I am going to Mumbai for my work @112242"
text_clean_obj = Text_Cleaning(text)
clean_text = text_clean_obj.clean_text(lemmatize_flag = False)
print(clean_text)
