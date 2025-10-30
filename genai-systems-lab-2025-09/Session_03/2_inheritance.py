class Animal:
    def speak(self):
        print("This animal makes a sound")

class Dog(Animal):
    def speak(self):   # override
        print("Woof!")

class Cat(Animal):
    def speak(self):
        print("Meow!")

d = Dog()
c = Cat()

d.speak()   # Woof!
c.speak()   # Meow!
a = Animal()
a.speak()   # This animal makes a sound

#=======================================================================
# Another Example
class GenAIModel:
    def generate(self, text):
        print("Generic model generating response...")

class OpenAIModel(GenAIModel):
    def generate(self, text):
        print(f"OpenAI GPT responding to: {text}")

class AnthropicModel(GenAIModel):
    def generate(self, text):
        print(f"Claude responding thoughtfully to: {text}")

models = [OpenAIModel(), AnthropicModel()]
for m in models:
    m.generate("What is an embedding?")
