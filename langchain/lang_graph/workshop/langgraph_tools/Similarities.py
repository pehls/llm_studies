from langchain_core.tools import tool

from typing import Annotated

import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

@tool
def cosine_similarity(
    sentence1: Annotated[str, "the first sentence to be calculated with cosine similarity"],
    sentence2: Annotated[str, "the second sentence to be calculated with cosine similarity"]
):
    """this is the cosine similarity tool.
    Use this tool to calculate the cosine similarity between two texts, or two sentences. it uses the cosine formula to return the similarity between two sentences
    Args:
        sentence1: first sentence as a string
        sentence2: second sentence as a string
    """
    from nltk.corpus import stopwords 
    from nltk.tokenize import word_tokenize
    from nltk.stem.snowball import SnowballStemmer
    try:
        list_sentence1 = word_tokenize(sentence1)
        list_sentence2 = word_tokenize(sentence2)
        # sw contains the list of stopwords 
        stemmer = SnowballStemmer('english', ignore_stopwords=True)
        l1 =[];l2 =[] 
        
        # remove stop words from the string 
        X_set = {stemmer.stem(w) for w in list_sentence1}  
        Y_set = {stemmer.stem(w) for w in list_sentence2} 
        
        # form a set containing keywords of both strings  
        rvector = X_set.union(Y_set)  
        for w in rvector: 
            if w in X_set: l1.append(1) # create a vector 
            else: l1.append(0) 
            if w in Y_set: l2.append(1) 
            else: l2.append(0) 
        c = 0
        
        # cosine formula  
        for i in range(len(rvector)): 
                c+= l1[i]*l2[i] 
        return (c / float((sum(l1)*sum(l2))**0.5)) 
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

@tool
def jaccard_similarity(
    sentence1: Annotated[str, "the first sentence to be calculated with jaccard similarity"],
    sentence2: Annotated[str, "the second sentence to be calculated with jaccard similarity"]
):
    """this is the jaccard similarity tool.
    Use this tool to calculate the jaccard similarity between two texts, or two sentences. it uses the jaccard formula to return the similarity between two sentences
    Args:
        sentence1: first sentence as a string
        sentence2: second sentence as a string
    """
    from nltk.corpus import stopwords 
    from nltk.tokenize import word_tokenize
    from nltk.stem.snowball import SnowballStemmer
    try:
        list_sentence1 = word_tokenize(sentence1)
        list_sentence2 = word_tokenize(sentence2)
        # sw contains the list of stopwords 
        stemmer = SnowballStemmer('english', ignore_stopwords=True)
        l1 =[];l2 =[] 
        
        # remove stop words from the string 
        X_set = {stemmer.stem(w) for w in list_sentence1}  
        Y_set = {stemmer.stem(w) for w in list_sentence2} 
        
        # form a set containing keywords of both strings  
        C = X_set.intersection(Y_set)  
        # form a set with a union
        D = X_set.union(Y_set)
        
        # jaccard formula  
        return float(len(C))/float(len(D))
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
@tool
def levenshtein_similarity(
    sentence1: Annotated[str, "the first sentence to be calculated with levenshtein similarity, or Levenshtein edit-distance between two strings"],
    sentence2: Annotated[str, "the second sentence to be calculated with levenshtein similarity, or Levenshtein edit-distance between two strings"]
):
    """this is the WuPalmer similarity, or wordnet similarity tool.
    Use this tool to calculate thelevenshtein similarity, or Levenshtein edit-distance between two strings, or two sentences. 
    The edit distance is the number of characters that need to be substituted, inserted, or deleted, to transform s1 into s2. 
    For example, transforming “rain” to “shine” requires three steps, consisting of two substitutions and one insertion: “rain” -> “sain” -> “shin” -> “shine”. 
    These operations could have been done in other orders, but at least three steps are needed.
    Args:
        sentence1: first sentence as a string
        sentence2: second sentence as a string
    """
    from nltk.corpus import stopwords 
    from nltk.tokenize import word_tokenize
    from nltk.stem.snowball import SnowballStemmer
    from nltk.metrics.distance import edit_distance
    try:
        list_sentence1 = word_tokenize(sentence1)
        list_sentence2 = word_tokenize(sentence2)
        # sw contains the list of stopwords 
        stemmer = SnowballStemmer('english', ignore_stopwords=True)
        l1 =[];l2 =[] 
        
        # remove stop words from the string 
        X_set = " ".join([stemmer.stem(w) for w in list_sentence1]) 
        Y_set = " ".join([stemmer.stem(w) for w in list_sentence2])
        
        # return distance
        return edit_distance(X_set, Y_set, substitution_cost=1, transpositions=True)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"