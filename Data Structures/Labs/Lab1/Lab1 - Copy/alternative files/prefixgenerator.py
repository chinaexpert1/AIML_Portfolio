import random

# Define the operands and operators
operands = ['A', 'B', 'C']
operators = ['+', '-', '*', '/', '$']

# Define a function to generate a random expression
def random_prefix():
    # Choose a random operator
    operator = random.choice(operators)
    # If the operator is $, choose two random operands and generate a subexpression
    if operator == '$':
        operand1 = random.choice(operands)
        operand2 = random.choice(operands)
        subexpression = random_prefix()
        expression = [operator, operand1, operand2, subexpression]
    # Otherwise, choose two random operands and generate a simple expression
    else:
        operand1 = random.choice(operands)
        operand2 = random.choice(operands)
        expression = [operator, operand1, operand2]
    # Convert the expression to a string in prefix format
    prefix_expression = ''.join(expression)
    # Otherwise, return the expression
    return prefix_expression


# Generate and print 10 random expressions
for i in range(10):
    x = random_prefix()

    # If the expression is less than 5 characters long, recursively generate a new expression
    if len(x) < 6:
        x += random_prefix()
    print(x)
