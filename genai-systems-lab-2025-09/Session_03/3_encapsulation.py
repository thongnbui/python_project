
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance   # private attribute

    def deposit(self, amount):
        self.__balance += amount

    def get_balance(self):
        return self.__balance

account = BankAccount(1000)
print(account.get_balance())   #  Works
account.deposit(500)
print(account.get_balance())   #  Works
# print(account.__balance)            # AttributeError (private)
# print(account._BankAccount__balance) # Works, but breaks encapsulation (not recommended)


#=======================================================================
class PromptTemplate:
    def __init__(self, template):
        self.__template = template   # private (hidden)

    def format_prompt(self, user_input):
        return self.__template.replace("{query}", user_input)

    def get_template(self):   # controlled access
        return self.__template

template = PromptTemplate("You are a helpful AI. Answer: {query}")
print(f"Formatted Template: {template.format_prompt('What is LangChain?')}") 
# print(template.__template)  #  Not accessible directly
print(f"Template: {template.get_template()}")  #  Access via method