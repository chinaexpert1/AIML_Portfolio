"""Andrew Taylor
    atayl136
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
from Stacksource import createStack


# postfix converter function
def postfix(expression):
    stack = createStack()                       # creating a stack
    postfix = ''                                # building an output
    alphas = False                              # ensuring input still has letters and operators while not
    ops = False                                 # needing to halt the program
    errors = 0

    for i in reversed(expression):
            if i.isalpha():
               stackPush(stack, i)
               alphas = True

            elif i in '+-*/$':
                stack, operand1 = stackPop(stack)
                if isEmpty(stack) == False:
                    stack, operand2 = stackPop(stack)
                else:
                    message = f"Invalid syntax: {i}. The program is trying to pop from an empty stack."
                    errors += 1
                    return message, 1
                result = operand1 + operand2 + i
                stackPush(stack, result)
                ops = True
            else:
                errors += 1
                return  'Invalid character: ' + i, 1         # so the function can spot syntax errors

    while len(stack)>0:
        stack, pop = stackPop(stack)
        postfix += pop

    # this is output when there is not both letters and operators in the input
    if not alphas or not ops:
       postfix = "Need both operands and operators in input."

    return postfix, errors
