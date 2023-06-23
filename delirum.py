import csv
import random
import statistics
from nltk import word_tokenize
from sklearn.linear_model import LinearRegression

# Path to the input CSV file
input_file = 'prompts.csv'

# Path to the output CSV file
output_file = 'worse_prompts.csv'

# Function to generate new, worse prompts based on existing prompts
def generate_prompt(existing_prompt):
    # Add your custom logic to generate new, worse prompts here
    # This is a simple example that shuffles the words in the prompt
    words = word_tokenize(existing_prompt)
    random.shuffle(words)
    new_prompt = ' '.join(words)
    return new_prompt

# Function to calculate the standard deviation between two sets of prompts
def calculate_standard_deviation(prompts1, prompts2):
    all_prompts = prompts1 + prompts2
    prompt_lengths = [len(prompt.split()) for prompt in all_prompts]
    return statistics.stdev(prompt_lengths)

# Read the input CSV file
with open(input_file, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    good_prompts = [row['prompt'] for row in reader]

# Generate new, worse prompts based on existing prompts
generated_prompts = []
for prompt in good_prompts:
    worse_prompt = generate_prompt(prompt)
    generated_prompts.append(worse_prompt)

# Train a linear regression model on the good prompts
regression_model = LinearRegression()
X = [[len(prompt.split())] for prompt in good_prompts]
y = [1] * len(good_prompts)  # Desired label (1 for good prompts)
regression_model.fit(X, y)

# Generate new, worse prompts based on the regression model
worse_prompts = []
for prompt in good_prompts:
    prompt_length = len(prompt.split())
    predicted_label = regression_model.predict([[prompt_length]])[0]
    if predicted_label >= 0.5:  # Custom threshold for worse prompts
        worse_prompt = generate_prompt(prompt)
        worse_prompts.append(worse_prompt)

# Calculate the standard deviation between the good prompts and generated worse prompts
std_dev = calculate_standard_deviation(good_prompts, worse_prompts)

# Write the generated worse prompts to the output CSV file
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['generated_prompt'])
    writer.writerows([[prompt] for prompt in worse_prompts])

print("Generated prompts saved to", output_file)
print("Standard Deviation:", std_dev)