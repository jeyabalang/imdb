import requests, nltk
nltk.download('punkt')
from bs4 import BeautifulSoup

# Make a request
page = requests.get(
    "https://www.imdb.com/title/tt0120338/reviews?ref_=tt_ov_rt")
soup = BeautifulSoup(page.content, 'html.parser')

# Extract title of page
page_title = soup.title.text

# Extract body of page
page_body = soup.body

# Extract head of page
page_head = soup.head

# print the result
container = soup.find_all(class_ = 'text show-more__control')
print(container)

sent_tokenized = [sent for sent in nltk.sent_tokenize(str(container))]

# Word Tokenize first sentence from sent_tokenized, save as words_tokenized
words_tokenized = [word for word in nltk.word_tokenize(sent_tokenized[0])]

# Remove tokens that do not contain any letters from words_tokenized
import re

filtered = [word for word in words_tokenized if re.search('[a-zA-Z]', word)]

# Display filtered words to observe words after tokenization
print(filtered)

# Import the SnowballStemmer to perform stemming
from nltk.stem.snowball import SnowballStemmer

# Create an English language SnowballStemmer object
stemmer = SnowballStemmer("english")

# Print filtered to observe words without stemming
print("Without stemming: ", filtered)

# Stem the words from filtered and store in stemmed_words
stemmed_words = [stemmer.stem(word) for word in filtered]

# Print the stemmed_words to observe words after stemming
print("After stemming:   ", stemmed_words)

# Define a function to perform both stemming and tokenization
def tokenize_and_stem(text):

    # Tokenize by sentence, then by word
    tokens = [y for x in nltk.sent_tokenize(text) for y in nltk.word_tokenize(x)]

    # Filter out raw tokens to remove noise
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]

    # Stem the filtered_tokens
    stems = [stemmer.stem(word) for word in filtered_tokens]

    return stems

words_stemmed = tokenize_and_stem(str(container))
print(words_stemmed)

# Import TfidfVectorizer to create TF-IDF vectors
from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiate TfidfVectorizer object with stopwords and tokenizer
# parameters for efficient processing of text
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features=200000,
                                 min_df=0.0, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem,
                                 ngram_range=(1,3))


tfidf_matrix =  tfidf_vectorizer.fit_transform([x for x in str(container)])
print(tfidf_vectorizer.get_feature_names())
print(tfidf_matrix.shape)

# Import k-means to perform clusters
from sklearn.cluster import KMeans

# Create a KMeans object with 5 clusters and save as km
km = KMeans(n_clusters=3)

# Fit the k-means object with tfidf_matrix
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()




# importing pandas as pd
import pandas as pd

# Create the dataframe
df = pd.DataFrame({'word':words_stemmed})

# Print the dataframe
print(df[:10])

print(tfidf_vectorizer.get_feature_names())



# # Create a column cluster to denote the generated cluster for each movie

print(df[:20],clusters[:20])
# # Display number of films per cluster (clusters from 0 to 4)
# movies_df['cluster'].value_counts()