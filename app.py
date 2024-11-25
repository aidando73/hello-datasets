from datasets import load_dataset
import matplotlib.pyplot as plt
from collections import Counter


# Load the dataset
data = load_dataset("meta-llama/Llama-3.2-1B-Instruct-evals", name="Llama-3.2-1B-Instruct-evals__gpqa__details")

output_prediction_texts = [str(example['output_prediction_text']) for example in data['latest']]

counter = Counter(output_prediction_texts)

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.bar(counter.keys(), counter.values())
plt.xlabel('output_prediction_text')
plt.ylabel('Frequency')
plt.title('Histogram of output_prediction_text from Llama-3.2-1B-Instruct-evals__gpqa__details')
plt.xticks(rotation=90)
plt.savefig('histogram.png')