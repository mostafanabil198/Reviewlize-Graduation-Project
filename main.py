#main.py

# Import the Flask module that has been installed.
from flask import Flask, jsonify, request

from evaluate import interactive_shell

from aop import reviewsAnalyzer

from textblob import TextBlob

import re

import string
import nltk
sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()

# Creating a new "app" by using the Flask constructor. Passes __name__ as a parameter.
app = Flask(__name__)


my_file = open("seed_list_n.txt", "r")
negative = my_file.read().splitlines()

my_file = open("seed_list_p.txt", "r")
positive = my_file.read().splitlines()


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '#', '—–']

# Annotation that allows the function to be hit at the specific URL.
@app.route("/")
# Generic Python functino that returns "Hello world!"
def index():
    return "Hello world! from flask"





# Annotation that allows the function to be hit at the specific URL. Indicates a GET HTTP method.
@app.route("/predict", methods=["POST"])
# Function that will run when the endpoint is hit.
def predict_model():
    # ret = interactive_shell(request.args.get("sentence"))
    # print("hiiiiiiiiiiiiiiiiiiiii")
    # ret2 = reviewsAnalyzer(request.args.get("sentence"), False)
    

    sentences = request.json["sentences"]
    aspects = []
    new_sentences = []
    

    

    def clean_puncts(x):
        x = str(x)
        for punct in puncts:
            x = x.replace(punct, '')
        return x

    for s in sentences:
        s = s.replace(",", " ")
        s = s.replace(".", " ")
        s = re.sub(' +',' ',s)
        s = s.translate( str.maketrans( '', '', string.punctuation )).strip()
        s = s.lower()
        
        s = clean_puncts(s)
        s = re.sub(' +',' ',s)
        if len(s) == 0 or s == " ":
            continue
        new_sentences.append(s)
    
    sentences = new_sentences
    for s in sentences:
        

        ret = interactive_shell(s)
        
        text = ret[0].split()
        extractions = ret[1].split()
        for i in range(len(extractions)):
            if extractions[i] == "B-A":
                aspect = text[i]
                while i + 1 < len(extractions) and extractions[i + 1] == "I-A":
                    aspect = aspect + " " + text[i + 1]
                    i += 1
                print("aspect model")
                print(aspect)
                print("******************************************************************")
                aspects.append(aspect)
    
    ret = reviewsAnalyzer(sentences, False)
    filtered_res = []
    for i in range (len(sentences)):
        sentence_res = ret[i]
        for j in range(len(sentence_res)):
            sentance_part = sentence_res[j]
            for part in sentance_part:
                if part[0] in aspects:
                     filtered_res.append(part)
                elif part[1] in aspects:
                    filtered_res.append([part[1], part[0]])
    aspect_opinion = []
    for pair in filtered_res:
        pol = TextBlob(pair[1])
        if pol.sentiment.polarity > 0:
            aspect_opinion.append([pair[0],"positive"])
        elif pol.sentiment.polarity <0:
            aspect_opinion.append([pair[0], "negative"])
        else:
            if pair[1] in positive:
                aspect_opinion.append([pair[0],"positive"])
            elif pair[1] in negative:
                aspect_opinion.append([pair[0], "negative"])
            else:
                aspect_opinion.append([pair[0], "nuteral"])
    unique_aspects = dict()
    index = 0
    for pair in aspect_opinion:
        if pair[0] not in unique_aspects:
            unique_aspects[pair[0]] = index
            index += 1
    positive_cnt = [0] * index
    negative_cnt = [0] * index
    neuteral_cnt = [0] * index
    for pair in aspect_opinion:
        i = unique_aspects[pair[0]]
        if pair[1] == "positive":
            positive_cnt[i] += 1
        elif pair[1] == "negative":
            negative_cnt[i] += 1
        else:
            neuteral_cnt[i] += 1
    
    aspects_count = dict()
    for key in unique_aspects:
        i = unique_aspects[key]
        aspects_count[key] = {"positive": positive_cnt[i], "negative": negative_cnt[i], "neutral": neuteral_cnt[i]}
    
    
    return jsonify({"results": {"aspects": aspects, "filtered": filtered_res, "aspect_analize": aspects_count}})






# Checks to see if the name of the package is the run as the main package.
if __name__ == "__main__":
    # Runs the Flask application only if the main.py file is being run.
    app.run()