import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
# Handle SSL Certificate Verification for NLTK on macOS
# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('stopwords')


# nltk.download('wordnet')
# nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()


stop_words = set(stopwords.words('english'))

# Load the dataset into a pandas DataFrame
file_path = './mbti_1.csv'

def load_data(file_path):
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
    print('\n')
    print('Cleaning data...')
    #removing urls
    df['Posts'] = df['Posts'].apply(lambda x: re.sub(r'http\S+', '', x))
    #converting to lowercase
    df['Posts'] = df['Posts'].apply(lambda x: x.lower())

    #remove punctuation, numbers and whitespace
    df['Posts'] = df['Posts'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df['Posts'] = df['Posts'].apply(lambda x: re.sub(r'\d+', '', x))
    df['Posts'] = df['Posts'].apply(lambda x: x.strip())
    df['Posts'] = df['Posts'].apply(lambda x: re.sub(r'\s+', ' ', x))

    # #removing stopwords
    stop_words = set(stopwords.words('english'))
    df['Posts'] = df['Posts'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    df['Posts'] = df['Posts'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))
    
    #lemmatizing is taking time
    df['Posts'] = df['Posts'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    print('Data cleaning completed')
    return df

valid_mbti_types = {'ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP',
                    'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'}

df = clean_data(load_data(file_path))
unique_types = df['Type'].unique()

# Print the unique values found in the 'Type' column
print("Unique MBTI Types in the DataFrame:")
print(unique_types)

x = df['Posts']
y = df['Type']
print('\n')
print('Splitting dataset into train-test...')  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
print("Splitting complete.")


def model_LR():
    print('\n')
    print('!-----!-----!-----!-----!-----!-----!-----!-----!-----!-----!-----!-----!')
    print('Applying LR model...')
    print('!-----!-----!-----!-----!-----!-----!-----!-----!-----!-----!-----!-----!')
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=2, max_features=10000, ngram_range=(1, 2))
    # Fit and transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print('\n')
    print('Loading model...')
    # Initialize Logistic Regression
    log_reg = LogisticRegression(C=1.0, max_iter=1000, solver='saga', penalty='l1')

    print("Model loaded.")

    # Train
    print('\n')
    print('Training...')
    log_reg.fit(X_train_tfidf, y_train)
    print('Trained.')

    # Predict and evaluate
    y_pred = log_reg.predict(X_test_tfidf)

    # Performance
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

print('\n')
model_LR()

