from transformers import AutoModel, AutoTokenizer

model_name = "hkunlp/instructor-xl"

# Download the model
model = AutoModel.from_pretrained(model_name)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to a local directory
model.save_pretrained("Model/")
tokenizer.save_pretrained("Model/")

# Assuming there is a class named HuggingFaceInstructEmbeddings
class HuggingFaceInstructEmbeddings:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

# Create an instance of the HuggingFaceInstructEmbeddings class
embeddings = HuggingFaceInstructEmbeddings(model=model, tokenizer=tokenizer)
