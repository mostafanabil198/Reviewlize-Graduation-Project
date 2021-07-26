# Reviewlize-Graduation-Project
 Reviewlize is a reviews analyzer web application that uses Aspect-Based Sentiment Analysis (ABSA) to extract product aspects and their polarities from the reviews.
 
 ---
## Review Analyzer module 
Extract the aspects in reviews, then extract the opinion related to each aspect. Finally, analyze the sentiment of each aspect.
  
  ### Citation
  Poria, S., Cambria, E. and Gelbukh, A., 2016. Aspect extraction for opinion mining with a deep convolutional neural network. Knowledge-Based Systems, 108, pp.42-49.

  ### Review Analizer Setup
  - sudo apt install python3
  - Pip3 install stanza
  - pip3 install flask
  - Pip3 install sklearn
  - Pip3 install textblob
  - Pip3 install tensorflow==1.5
  - corenlp_dir = './corenlp'
  - stanza.install_corenlp(dir=corenlp_dir)
  
  ### Running Model
  Download Glove embeddings (GloVe: http://nlp.stanford.edu/data/glove.840B.300d.zip )

  1. [DO NOT MISS THIS STEP] Build vocab from the data and extract trimmed glove vectors according to the config in `model/config.py`.

  ```
  python build_data.py
  ```

  2. Train the model with

  ```
  python train.py
  ```


  3. Evaluate and interact with the model with
  ```
  python evaluate.py
  ```
  ### Run Flask Server
  * python3 main.py
 ---



