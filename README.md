# Reviewlize-Graduation-Project
Reviewlize is a reviews analyzer web application that uses Aspect-Based Sentiment Analysis (ABSA) to extract product aspects and their polarities from the reviews.
# Review Analyzer module 
Extract the aspects in reviews, then extract the opinion related to each aspect. Finally, analyze the sentiment of each aspect.


# Flask Server Setup
- sudo apt install python3
- Pip3 install stanza
- pip3 install flask
- Pip3 install sklearn
- Pip3 install textblob
- Pip3 install tensorflow==1.5
- corenlp_dir = './corenlp'
- stanza.install_corenlp(dir=corenlp_dir)

# Run Flask Server
* python3 main.py


