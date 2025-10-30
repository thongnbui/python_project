class Car:
    def __init__(self, brand, model):
        self.brand = brand      # attribute
        self.model = model

    def display(self):          # method
        print(f"This car is a {self.brand} {self.model}")

# Creating objects
car1 = Car("Tesla", "Model S")
car2 = Car("Toyota", "Camry")

car1.display()
car2.display()

#=======================================================================
# Another Example
class LLM:
    def __init__(self, name, provider):
        self.name = name
        self.provider = provider

    def generate(self, prompt):
        print(f"[{self.provider}-{self.name}] Generating response for: {prompt}")

# Objects (instances of models)
gpt = LLM("GPT-4", "OpenAI")
llama = LLM("Llama-3", "Meta")

gpt.generate("Explain RAG in simple terms.")
llama.generate("Write Python code for a chatbot.")