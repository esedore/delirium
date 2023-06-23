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
    # Add  custom logic to generate new, worse prompts here
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
with open(input_file, 'r') as file:
    reader = csv.DictReader(file)
    good_prompts = [row['prompt'] for row in reader]

# Generate new, worse prompts based on existing prompts
generated_prompts = []
for prompt in good_prompts:
    worse_prompt = generate_prompt(prompt)
    generated_prompts.append(worse_prompt)

# Calculate the standard deviation between the good prompts and generated worse prompts
std_dev = calculate_standard_deviation(good_prompts, generated_prompts)

# Write the generated worse prompts to the output CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['generated_prompt'])
    writer.writerows([[prompt] for prompt in generated_prompts])

print("Generated prompts saved to", output_file)
print("Standard Deviation:", std_dev)