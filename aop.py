# Import stanza
import stanza
import os

import string

import nltk
sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()

corenlp_dir = './corenlp'
os.environ["CORENLP_HOME"] = corenlp_dir
# stanza.install_corenlp(dir=corenlp_dir)
from stanza.server import CoreNLPClient
    
# Construct a CoreNLPClient with some basic annotators, a memory allocation of 4GB, and port number 9001
client = CoreNLPClient(
        annotators=['tokenize','pos','lemma','depparse'],
        memory='16G',
        endpoint='http://localhost:9002',
        be_quiet=True)

print(client)

# Start the background server and wait for some time
# Note that in practice this is totally optional, as by default the server will be started when the first annotation is performed
client.start()

def reviewsAnalyzer(reviews_list, debug = False):
  # list to hold each review's sentences and its aspects 
  # reviews->[ sentences->[ aspects->[{aspect: opinion}, {aspect, opinion}]], [[]], [[]], ]
  reviews_sentences_aspects = []

  # For each review:
    # 1- split to sentences using spacy
    # 2- for each sentence:
      # 3- get dependences
      # 4- extract aspect_opinion pairs
  
  for review in reviews_list:
    # print("-=-=R-=>", review)
    # review_boundries = nlp(review)
    review_boundries = sent_tokenizer.tokenize(review)
    review_sentences = []
    for sentence in list(review_boundries): # 1, 2
      # print("-=-=S-=>", sentence)
      sentence_aspects = []
      sentence1 = str(sentence).translate( str.maketrans( '', '', string.punctuation )).strip()
      # print("-=-=S1-=>", sentence1)
      if (sentence1 == '' or sentence1 == ' ' or len(sentence1) == 0):
            continue
      list_dep, list_pos = getDependencies(sentence1) # 3
      aspect_opinion_pairs = aspectOpinionPairExtractor(list_dep=list_dep, list_pos=list_pos) # 4
      if debug == True:
        print("==========")
        print("sentence: ", sentence1)
        print("dependencies: ", list_dep)
        print("POS: ", list_pos)
        print("aspects opinion pairs: ", aspect_opinion_pairs)
        print("==========")
      for as_op_pair in aspect_opinion_pairs:
        sentence_aspects.append(as_op_pair)
      review_sentences.append(sentence_aspects)
    reviews_sentences_aspects.append(review_sentences)
  return reviews_sentences_aspects

def getDependencies(review_sentence):
  document = client.annotate(str(review_sentence))
  # print("=====>", review_sentence)
  sentence = document.sentence[0]
  dependency_parse = sentence.basicDependencies
  token_dict = {}
  pos_dict = {}
  for i in range(0, len(sentence.token)) :
      token_dict[sentence.token[i].tokenEndIndex] = sentence.token[i].word
      pos_dict[sentence.token[i].word] = sentence.token[i].pos

  # get a list of the dependencies with the words they connect
  list_dep=[]
  for i in range(0, len(dependency_parse.edge)):

      source_node = dependency_parse.edge[i].source
      source_name = token_dict[source_node]

      target_node = dependency_parse.edge[i].target
      target_name = token_dict[target_node]

      dep = dependency_parse.edge[i].dep
      list_dep.append((dep, source_name, target_name))
  # print(list_dep)
  return list_dep, pos_dict

def aspectOpinionPairExtractor(list_dep, list_pos):
  aspect_opinion_pairs = set()
  visited_dep = [False]*len(list_dep)
  for dep in list_dep:
    if  'nsubj' in dep[0]:
      # print("nsubj")
      if "JJ" in list_pos[dep[1]] and "NN" in list_pos[dep[2]]: #1
        aspect, opinion = nsubjAdj(list_dep, list_pos, dep)
        addPair(aspect, opinion, list_dep, aspect_opinion_pairs)
      elif "JJ" in list_pos[dep[1]] and "NN" not in list_pos[dep[2]]: #2
        aspect, opinion = apply_nsubj_second(list_dep, dep)
        addPair(aspect, opinion, list_dep, aspect_opinion_pairs)
      elif "VB" in list_pos[dep[1]] and "NN" not in list_pos[dep[2]]: #3
        aspect, opinion = opinionVerb(list_dep, dep)
        addPair(aspect, opinion, list_dep, aspect_opinion_pairs)
      elif "VB" in list_pos[dep[1]] and ("NN" in list_pos[dep[2]]): #4
        aspect, opinion = apply_nsubj_forth(list_dep, dep)
        addPair(aspect, opinion, list_dep, aspect_opinion_pairs)
      #elif "NN" in list_pos[dep[1]] and "NN" in list_pos[dep[2]]: #10
        #aspect, opinion = apply_nsubj_ten(list_dep, dep)
        #addPair(aspect, opinion, list_dep, aspect_opinion_pairs)
      elif "NN" in list_pos[dep[1]]: #10
        aspect, opinion = apply_nsubj_ten(list_dep, dep)
        addPair(aspect, opinion, list_dep, aspect_opinion_pairs)
    elif dep[0] == 'amod' and visited_dep[list_dep.index(dep)] == False:
      # print("amod")
      if "JJ" in list_pos[dep[2]]:
        aspect, opinion = apply_amod_sixth(list_dep, dep) #6
        # Get multi aspects for same opinion
        aspects = apply_amod_eight(list_dep, aspect) #8
        aspects.append(aspect)
        for a in aspects:
          addPair(a, opinion, list_dep, aspect_opinion_pairs)

        # Get multi opinions for same aspect
        aspect, opinions = multiOpinion(list_dep, dep, visited_dep) #7
        for o in opinions:
          addPair(aspect, o, list_dep, aspect_opinion_pairs)
    elif dep[0] == 'obl' and "JJ" in list_pos[dep[1]]: #9
      # print("obl")
      aspect = dep[2]
      opinion = dep[1]
      addPair(aspect, opinion, list_dep, aspect_opinion_pairs)
    elif dep[0] == 'acl:relcl': #5
      # print("acl:relcl")
      aspect, opinion = relativeClause(list_dep, dep)
      addPair(aspect, opinion, list_dep, aspect_opinion_pairs)
    elif 'nmod' in dep[0]: #new
      aspect = dep[1]
      opinion = dep[2]
      addPair(aspect, opinion, list_dep, aspect_opinion_pairs)
  return aspect_opinion_pairs

def addPair(aspect, opinion, list_dep, aspect_opinion_pairs):
  if aspect == "" or opinion == "":
    return
  aspect = getCompund(list_dep, aspect)
  opinion = check_neg(list_dep, opinion)
  p = (aspect, opinion)
  # print(p)
  aspect_opinion_pairs.add(p)

# Call after any function to return compund aspects or same aspect => battery life
# list_dep => dependency_parsed
# aspect => single aspect "String"
def getCompund(list_dep, aspect):
  for dep in list_dep:
    if dep[0] == "compound" and dep[1] == aspect :
      aspect = dep[2] + " " + aspect
      break;
  return aspect

# 1
# Extract pair when it's after capular verb => the battery is good 
# and if having comparitave adj then get the advcl => the battery is more satisfying
# list_dep => dependency_parsed
# list_pos => Part Of Speech "dictionary"
# dependency => the nsubj dep that leaded to this case
def nsubjAdj(list_dep, list_pos, dependency):
  aspect = dependency[2]
  opinion = dependency[1]
  if list_pos[dependency[1]] == "JJR":
    for dep in list_dep:
      if dep[0] == "advcl" and dep[1] == dependency[1]:
        opinion = dep[2]
  return aspect, opinion

# 2
# when the second argument in nsubj dependency is not noun and the first is adj.
# the feature of the sentence will be the object of it.
# use dependencies (nsubj , xcomp , obj) to get the feature.
# Ex : it is great having the LCD Display.
def apply_nsubj_second(list_dep, dependancy):
  f_arg = ''
  s_arg = dependancy[1]
  verb = ''
  for dep in list_dep:
    if dep[0] == 'xcomp' and dep[1] == s_arg :
      verb = dep[2]
      break
    if dep[0] == 'dep' and dep[1] == s_arg :
      verb = dep[2]
      break
  for dep in list_dep:
    if dep[0] == 'obj' and dep[1] == verb :
      f_arg = dep[2]
      break
  # for dep in list_dep:
  #   if dep[0] == 'compound' and dep[1] == f_arg :
  #     f_arg = dep[2] + " "+f_arg
  #     break
  # print(s_arg)
  # print(f_arg)
  return f_arg , s_arg

# 3
# Extract pair when the opinions is the verb like love and like
# list_dep => dependency_parsed
# list_pos => Part Of Speech "dictionary"
# dependency => the current dep that leaded to this case
# sentiment_verbs => list of verbs that may be opinion
sentiment_verbs = ["like", "love", "adore", "enjoyed", "liked", "loved", "enjoy", "overloaded"]
def opinionVerb(list_dep, dependency):
  aspect = ""
  opinion = ""
  # if(sentiment_verbs.count(dependency[1])):
  opinion = dependency[1]
  for dep in list_dep:
    if ("obj" in dep[0] and dep[1] == opinion):
      aspect = dep[2]
  return aspect, opinion


# 4
# when the second argument is noun and the opnion part is verb.
# the opinion will be the complement of the verb.
# Ex : the flash works great.
def apply_nsubj_forth (list_dep, dependency) :
  f_arg = dependency[2]
  s_arg = ''
  verb = dependency[1]
  for dep in list_dep:
    if dep[0] == 'xcomp' and dep[1] == verb :
      s_arg = dep[2]
      break
    if dep[0] == 'advmod' and dep[1] == verb :
      s_arg = dep[2]
      break
  # for dep in list_dep:
  #   if dep[0] == 'compound' and dep[1] == f_arg :
  #     f_arg = dep[2] + " "+f_arg
  #     break
  # print(f_arg)
  # print(s_arg)
  return f_arg , s_arg

# 5
# Extract pair when there is a relative clause relation between the aspect and the opinion => movie mode that works good
# list_dep => dependency_parsed
# dependency => the current dep that leaded to this case
def relativeClause(list_dep, dependency):
  aspect = dependency[1]
  relcl_opinion = dependency[2]
  opinion = ""
  for dep in list_dep:
    if dep[0] == "advmod" and relcl_opinion == dep[1]:
      opinion = dep[2]
      return aspect, opinion
  return aspect, opinion

# 6
# when the feature and opinion pair candidate could in same dependency.
# amod dependency the first arg is considered to be the feature.
# the second arg is considered to be the opinion words.
# Ex : this is a great screen.
def apply_amod_sixth (list_dep, dependency) :
  f_arg = dependency[1]
  s_arg = dependency[2]
  # for dep in list_dep:
  #   if dep[0] == 'compound' and dep[1] == f_arg :
  #     f_arg = dep[2] + " "+f_arg
  #     break
  # print(f_arg)
  # print(s_arg)
  return f_arg , s_arg

# 7-1
# Extract multi opinions if without "and" word. only with ","
# list_dep => dependency_parsed
# dependency => the current dep that leaded to this case
# visited_dep => to make sure if the amod dependency was considered or not in the main loop
def multiOpinion(list_dep, dependency, visited_dep):
  aspect = dependency[1]
  opinions = []
  for dep in list_dep:
    if(dep[0] == "amod" and dependency[1] == dep[1]):
      opinions.append(dep[2])
      visited_dep[list_dep.index(dep)] = True
  opinions = getAndOpinions(list_dep, opinions)
  return aspect, opinions

# 7-2
# Call after any function to get list of opinions if multi exist with "and" word
# list_dep => dependency_parsed
# opnions => list of opinions "List"
def getAndOpinions(list_dep, opinions):
  for dep in list_dep:
    if dep[0] == "conj" and (opinions.count(dep[1]) != 0):
      opinions.append(dep[2])
    elif dep[0] == "conj" and (opinions.count(dep[2]) != 0):
      opinions.append(dep[1])
  return opinions

# 8
# when the same opinion word is used to describe more than one feature.
# apply conj dependency on the related features. 
# Ex : Nokia has a good screen and battery.
def apply_amod_eight (list_dep, feature) :
  feats = []
  for dep in list_dep:
    if dep[0] == 'conj' and dep[1] == feature:
      feats.append(dep[2])
  #call the function check compound of each feature
  return feats

# 9
# Extract pair when theres a preposition => i am happy "with" my phone
# dependency => the current dep that leaded to this case
def prepositions(dependency):
  aspect = dependency[2]
  opinion = dependency[1]
  return aspect, opinion

# 10
# when nsubj dependency is between two nouns.
# the first argument can be used as a opinion word.
# Ex : the battery was never a problem.
def apply_nsubj_ten (list_dep, dependency):
  f_arg = dependency[2]
  s_arg = dependency[1]
  # for dep in list_dep:
  #   if dep[0] == 'compound' and dep[1] == f_arg :
  #     f_arg = dep[2] + " "+f_arg
  #     break
  # print(s_arg)
  # print(f_arg)
  return f_arg , s_arg

# 11
# Call after any function to get list of aspects if multi exist
# list_dep => dependency_parsed
# aspect => single aspect "String"
def getAndAspects(list_dep, aspect):
  aspects = [aspect]
  for dep in list_dep:
    if dep[0] == "conj" and (dep[1] == aspect):
      aspects.append(dep[2])
    elif dep[0] == "conj" and (dep[2] == aspect):
      aspects.append(dep[1])
  for i in range(len(aspects)):
    aspects[i] = getCompund(list_dep, aspects[i])
  return aspects

# neg
# check negation of the opinion words as it can reverse the sentiment.
def check_neg(list_dep, adj):
    list_neg = ['no' , 'never' , 'not' , 'n\'t' , 'none' , 'neither'] 
    #can also check the negative seed list for that.
    for dep in list_dep:
      if dep[0] == 'advmod' and dep[1] == adj and ( dep[2] in list_neg) :
        adj = dep[2] + ' ' + adj
        return adj
    return adj

