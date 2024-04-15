import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


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
file_path = '/Users/admin/NLP_assignment3/mbti_1.csv'

def load_data(file_path):
    types = []
    posts = []

    # Read the file manually
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Assume each line is of at least length 4
                if len(line) > 4:
                    types.append(line[:4])  # First 4 characters
                    posts.append(line[5:].strip())  # Rest of the line
                    
        # Create DataFrame from lists
        df = pd.DataFrame({
            'Type': types,
            'Posts': posts
        })
        print("Data loaded and columns split successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return df

def clean_data(df):
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
    # df['Posts'] = df['Posts'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    return df
df = clean_data(load_data(file_path))

x = df['Posts'] 
y = df['Type']   


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# Output the results to check
print("Training set:")
print(X_train.head(), y_train.head())
print("\nTest set:")
print(X_test.head(), y_test.head())