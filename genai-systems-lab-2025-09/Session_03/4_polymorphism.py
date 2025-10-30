class Bird:
    def fly(self):
        print("Some birds can fly")

class Sparrow(Bird):
    def fly(self):
        print("Sparrow flies high")

class Penguin(Bird):
    def fly(self):
        print("Penguins cannot fly")

for bird in [Sparrow(), Penguin()]:
    bird.fly()

#=======================================================================
# Another Example
class EmbeddingModel:
    def embed(self, text):
        pass

class OpenAIEmbedding(EmbeddingModel):
    def embed(self, text):
        print(f"OpenAI embedding vector created for: {text}")

class HuggingFaceEmbedding(EmbeddingModel):
    def embed(self, text):
        print(f"HuggingFace embedding vector created for: {text}")

for model in [OpenAIEmbedding(), HuggingFaceEmbedding()]:
    model.embed("Generative AI use cases")
