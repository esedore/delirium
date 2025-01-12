import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.tokenize import word_tokenize
from textstat import flesch_reading_ease
from sklearn.preprocessing import StandardScaler
import nltk
import logging
from nltk.corpus import wordnet

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Paths
input_file = "prompts.csv"  # Large dataset
reduced_file = "small_prompts.csv"  # Reduced dataset
degraded_output_file = "degraded_prompts.csv"  # Poor-quality prompts

# Parameters
chunk_size = 100000
target_size = 100000
test_size = 0.2
random_state = 42

# Step 1: Reduce the dataset
def reduce_dataset(input_file, target_size):
    try:
        df = pd.read_csv(input_file)
        df = df[df["prompt"].str.len() > 10]  # Filter rows with prompts longer than 10 characters
        reduced_data = df.sample(n=target_size, random_state=random_state)
        reduced_data.to_csv(reduced_file, index=False)
        logging.info(f"Reduced dataset saved to {reduced_file} with {len(reduced_data)} rows.")
        return reduced_data
    except FileNotFoundError:
        logging.error("Input file not found.")
        exit()

# Step 2: Extract features
def extract_features(df):
    df["length"] = df["prompt"].str.len()  # Length of prompt
    df["word_count"] = df["prompt"].str.split().str.len()  # Word count
    df["readability"] = df["prompt"].map(flesch_reading_ease)  # Readability score
    return df[["length", "word_count", "readability"]]

# Step 3: Train the model
def train_model(data, test_size, random_state):
    try:
        data["label"] = data["prompt"].apply(lambda x: 1 if len(x.split()) > 5 else 0)
        X = extract_features(data)
        y = data["label"]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # Standardize features
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        logging.info("Model Performance:")
        logging.info(f"Accuracy: {accuracy_score(y_val, y_pred)}")
        logging.info(f"\n{classification_report(y_val, y_pred)}")
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
        return model, scaler
    except Exception as e:
        logging.error(f"Error training the model: {e}")
        exit()

# Step 4: Generate degraded prompts
def synonym_replace(word):
    synonyms = wordnet.synsets(word)
    if synonyms:
        return synonyms[0].lemmas()[0].name()  # Return the first synonym
    return word

def degrade_prompt(prompt):
    noise_words = ["foo", "bar", "nonsense", "baz", "qux", "irrelevant"]
    noise_phrases = [
        "undefined behavior", "bad data ahead", "this makes no sense",
        "unexpected token", "random chaos incoming", "irrelevant nonsense detected"
    ]
    contradictory_terms = {"detailed": "blurry", "sharp": "vague", "beautiful": "ugly", "elegant": "rough"}

    words = word_tokenize(prompt)

    # Replace words with synonyms
    words = [synonym_replace(w) if random.random() < 0.2 else w for w in words]

    # Swap two random words
    if len(words) > 1:
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]

    # Insert noise at random positions
    for _ in range(random.randint(1, 3)):  # Insert 1-3 noise elements
        random_idx = random.randint(0, len(words) - 1)
        words.insert(random_idx, random.choice(noise_words + noise_phrases))

    # Replace specific keywords with contradictory terms
    for key, value in contradictory_terms.items():
        if key in words:
            words[words.index(key)] = value

    # Add random punctuation
    if len(words) > 5:
        words[random.randint(0, len(words) - 1)] += random.choice(["!", "?", ",,,"])

    # Randomly delete words
    if len(words) > 5:
        for _ in range(random.randint(1, 2)):  # Delete 1-2 words
            del words[random.randint(0, len(words) - 1)]

    return " ".join(words)

def generate_degraded_prompts(data, model, scaler):
    try:
        X_test = extract_features(data)
        X_test = scaler.transform(X_test)  # Standardize features
        data["predicted_label"] = model.predict(X_test)
        data["degraded_prompt"] = data["prompt"].apply(degrade_prompt)
        data.to_csv(degraded_output_file, index=False)
        logging.info(f"Degraded prompts saved to {degraded_output_file}")
    except Exception as e:
        logging.error(f"Error generating degraded prompts: {e}")
        exit()

# Main script execution
if __name__ == "__main__":
    logging.info("Reducing dataset...")
    reduced_data = reduce_dataset(input_file, target_size)

    logging.info("Training model...")
    model, scaler = train_model(reduced_data, test_size, random_state)

    logging.info("Generating degraded prompts...")
    generate_degraded_prompts(reduced_data, model, scaler)