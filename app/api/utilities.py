import re
import pickle

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

with open('models/pipeline.pickle', 'rb') as f:
    loaded_pipe = pickle.load(f)


def predict_pipeline(text):
    output = predict(loaded_pipe, text)
    return output

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

## Defining set containing all stopwords in english.
stopwords = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

def preprocess(textdata):
    processed_text = []

    url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    user_pattern = '@[^\s]+'
    alpha_pattern = "[^a-zA-Z0-9]"
    sequence_pattern = r"(.)\1\1+"
    seq_replace_pattern = r"\1\1"

    for tweet in textdata:
        tweet = tweet.lower()
        
        tweet = re.sub(url_pattern, ' URL', tweet)
        
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
        
        tweet = re.sub(user_pattern, " USER", tweet)
        
        tweet = re.sub(alpha_pattern, " ", tweet)
        
        tweet = re.sub(sequence_pattern, seq_replace_pattern, tweet)
        
        preprocessed_words = []
        
        for word in tweet.split():
            if len(word) > 1 and word not in stopwords:
                word = lemmatizer.lemmatize(word)
                preprocessed_words.append(word)
                
        processed_text.append(' '.join(preprocessed_words))

    return processed_text

def predict(model, text):
    preprocessed_text = preprocess(text)
    predictions = model.predict(preprocessed_text)
    
    pred_to_label = {0: 'Negative', 1: 'Positive'}
    
    
    data = []
    for t, pred in zip(text, predictions):
        data.append({'text': t, 'pred':int(pred), 'label':pred_to_label[pred]})
        
    return data

if __name__ == "__main__":
    text = ["I hate twitter",
           "May the force be with you.",
           "Elon Musk is a silly silly man for trashing this site."]
    
    predictions = predict_pipeline(text)
    print(predictions)