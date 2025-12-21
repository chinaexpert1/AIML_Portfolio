"""
Converts a prefix expression to postfix notation.

Args:
expression (str): A string representing the prefix expression to be converted.

Returns:
str: A string representing the postfix notation of the input prefix expression.

Example:
>>> expression = "* + 2 3 4"
>>> prefix_to_postfix(expression)
'2 3 + 4 *'

Note:
- The supported operators are: '+', '-', '*', '/', and '$'.
- The function assumes that the input expression is a valid prefix expression, where the operators appear before their operands.
- The function uses a stack to convert the prefix expression to postfix notation.

This is a simple program consisting of one function so it was encapsulated in one file for simplicity.
"""

import Stacksource

# postfix converter function
def postfix(expression):
    stack = []
    postfix = ''
    for i in reversed(expression):
        try:
            if i.isalpha():
                stack.append(i)
            elif i in '+-*/$':
                operand1 = stack.pop()
                operand2 = stack.pop()
                result = operand1 + operand2 + i
                stack.append(result)
            else:
                raise ValueError                      # ValueError included here also
        except:                                       # so the function can spot syntax errors
            return 'Invalid syntax or character: ' + i             # and still used as intended
    while stack:
        postfix += stack.pop()
    return postfix

#Test code
#expression = '*+ABC'
#result = postfix(expression)
#print('Postfix expression: ', result)
