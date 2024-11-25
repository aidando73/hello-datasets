import datasets

# Load the dataset
data = datasets.load_dataset("meta-llama/Llama-3.2-1B-Instruct-evals", name="Llama-3.2-1B-Instruct-evals")

# Print the first 5 examples
print(data["latest"][:5])