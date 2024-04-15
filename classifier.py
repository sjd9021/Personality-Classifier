import pandas as pd

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

df = load_data(file_path)
print(df.head())