import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):  # Check for NaN or non-string values
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    return ' '.join(tokens)

# Load dataset
input_file_path = r"D:\Ichrak\SUPCOM\AI Project\Dirty Dataset\True.csv"  # Path to the input CSV
df = pd.read_csv(input_file_path)

# Apply cleaning to the relevant columns
columns_to_clean = ['title', 'subject', 'date', 'text']  # Columns to clean
for col in columns_to_clean:
    if col in df.columns:  # Ensure column exists in the DataFrame
        df[f'cleaned_{col}'] = df[col].apply(clean_text)

# Save only the cleaned columns to a new CSV file
cleaned_columns = [f'cleaned_{col}' for col in columns_to_clean if f'cleaned_{col}' in df.columns]

# Specify the output directory and ensure it exists
output_dir = r"D:\Ichrak\SUPCOM\AI Project\Cleaned Dataset"  # Path to save the cleaned dataset
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save the cleaned dataset to the specified path
output_file_path = os.path.join(output_dir, "True data (cleaned).csv")
df[cleaned_columns].to_csv(output_file_path, index=False)

print(f"Cleaned data saved to '{output_file_path}'")
