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

from Stacksource import *


# postfix converter function
def postfix(expression):
    stack = []
    postfix = ''
    alphas = False
    ops = False
    for i in reversed(expression):
       # try:
            if i.isalpha():
               stackPush(stack, i)
               alphas = True

            elif i in '+-*/$':
                stack, operand1 = stackPop(stack)
                print(stack)
                print(type(operand1))
                if isEmpty(stack) == True:  # it was necessary to check again here for an empty stack
                    return 'Invalid syntax or character: ' + i
                stack, operand2 = stackPop(stack)
                print(type(operand2))
                if not (type(operand1) or type(operand2)) != None:
                    result = operand1 + operand2 + i
                    stackPush(stack, result)
                ops = True

        #    else:
        #        raise ValueError                      # ValueError included here also
        #except:                                       # so the function can spot syntax errors
         #   return 'Invalid syntax or character: ' + i             # and still used as intended

    while len(stack)>0:
        stack, pop = stackPop(stack)
        postfix += pop

    # this is output when there is not both letters and operators in the input
    if not alphas or not ops:
       postfix = "Need both operands and operators in input."

    return postfix

#Test code
expression = '*+ABC'
result = postfix(expression)
print('Postfix expression: ', result)


