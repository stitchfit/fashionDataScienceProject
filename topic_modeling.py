import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis.gensim

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('fashiondata.csv')

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return tokens
    else:
        return []  # Return an empty list for non-string values

# Preprocessing the "Hashtags" column
df['processed_hashtags'] = df['Hashtags'].apply(preprocess_text)

# Preprocessing the "Caption" column
df['processed_caption'] = df['Caption'].apply(preprocess_text)

# Combine the processed hashtags and captions into a single list for each row
df['combined_text'] = df.apply(lambda row: row['processed_hashtags'] + row['processed_caption'], axis=1)

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(df['combined_text'])

# Create a document-term matrix
corpus = [dictionary.doc2bow(doc) for doc in df['combined_text']]

# Build the LDA model
num_topics = 5
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Create the visualization
vis_data = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)

# Save the visualization as an HTML file
pyLDAvis.save_html(vis_data, 'topic_modeling_visualization.html')

# Visualization
pyLDAvis.display(vis_data)
