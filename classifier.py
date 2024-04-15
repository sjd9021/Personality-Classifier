import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# Initialize the lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Path to the dataset
file_path = './mbti_1.csv'

# List of valid MBTI types
valid_mbti_types = {
    'ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP',
    'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'
}

def load_data(file_path):
    """ Load data from a CSV file, filtering for valid MBTI types. """
    print('\nLoading data from csv...')
    types = []
    posts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if len(line) > 4:
                    type_label = line[:4].strip()  # Ensure no extra whitespace
                    if type_label in valid_mbti_types:  # Check if it's a valid MBTI type
                        types.append(type_label)
                        posts.append(line[5:].strip())  # Rest of the line
        df = pd.DataFrame({'Type': types, 'Posts': posts})
        print("Data loaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return df

def clean_data(df):
    """ Clean text data in the dataframe. """
    print('\nCleaning data...')
    df['Posts'] = df['Posts'].apply(lambda x: re.sub(r'http\S+', '', x))  # Removing URLs
    df['Posts'] = df['Posts'].apply(lambda x: x.lower())  # Converting to lowercase
    df['Posts'] = df['Posts'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove punctuation
    df['Posts'] = df['Posts'].apply(lambda x: re.sub(r'\d+', '', x))  # Remove digits
    df['Posts'] = df['Posts'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())  # Remove extra spaces
    df['Posts'] = df['Posts'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))  # Removing stopwords
    df['Posts'] = df['Posts'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))  # Lemmatizing
    print('Data cleaning completed')
    return df

df = clean_data(load_data(file_path))

x = df['Posts']
y = df['Type']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.80, random_state=100)
print("\nSplitting complete.")

def model_LR():
    """ Train and evaluate a logistic regression model. """
    print('\nApplying LR model...')
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=2, max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)  # Transform training data
    X_test_tfidf = vectorizer.transform(X_test)  # Transform test data

    print('\nLoading model...')
    log_reg = LogisticRegression(C=1.0, max_iter=1000, solver='saga', penalty='l1')
    print("Model loaded.")

    print('\nTraining...')
    log_reg.fit(X_train_tfidf, y_train)
    print('Trained.')

    y_pred = log_reg.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

model_LR()
