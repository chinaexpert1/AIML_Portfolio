"""Andrew Taylor
    atayl136
Converts a prefix expression to postfix notation.

Args:
expression (str): A string representing the prefix expression to be converted.

Returns:
str: A string representing the postfix notation of the input prefix expression.

Example:
>>> expression = "*+ABC"
>>> postfix(expression)
'AB+C*'

Note:
- The supported operators are: '+', '-', '*', '/', and '$'.
- The function uses a stack to convert the prefix expression to postfix notation.
- The stack functions are imported from a stack source code file
"""

# Import Statement - Stack Source Code
from Stacksource import *


# Postfix Converter Function Below
# The function makes use of the functions defined in the Stacksource module to create and manipulate a stack that
# converts prefix to postfix expressions. Care is taken to avoid errors that would occur from erroneous input passed in,
# or no input at all. Errors are counted for reporting later.
def postfix(expression):
    stack = createStack()                       # creating a stack
    postfix = ''                                # building an output
    alphas = False                              # ensuring input still has letters and operators while not
    ops = False                                 # needing to halt the program
    errors = 0                                  # keeping a binary value for an error counter external to the function

    for i in reversed(expression):              # prefix expressions are read right to left
            if i.isalpha():                     # only letters as operands are recognized
               stackPush(stack, i)              # and they are pushed to the stack
               alphas = True                    # then noted as present

            elif i in '+-*/$':                                  # 5 operator types are valid
                if isEmpty(stack) == False:                     # the empty stack checking function is necessary
                    stack, operand1 = stackPop(stack)           # to avoid errors with popping. popping occurs here
                if isEmpty(stack) == False:                     # in accordance with the conversion algorithm.
                    stack, operand2 = stackPop(stack)
                else:
                    message = f"Invalid syntax: {i}. The program is trying to pop from an empty stack."
                    errors += 1                                 # the alternative is an empty stack error, it is counted
                    return message, 1
                result = operand1 + operand2 + i                # the algorithm works by pairing up operands, and buckets
                stackPush(stack, result)                        # of operands which are then pushed to the stack
                ops = True                                      # operators can now be noted as present
            else:
                errors += 1                                  # this alternative is that the character is invalid
                return  'Invalid character: ' + i, 1         # so the function can handle input that has errors.
                                                             # I made this choice for the integrity of the final report
                                                             # and the data flow internal to the program. I wanted my
                                                             # converter to be able to handle that possiblity even though
                                                             # the input has been screened. I felt that prudent.
    while len(stack)>0:
        stack, pop = stackPop(stack)                         # when the expression has been processed fully the stack is
        postfix += pop                                       # popped until it is empty, and the contents are joined
                                                             # this is the rule that completes the algorithm.

    # this is the output message  when there is not both letters and operators in the input
    if not alphas or not ops:
       postfix = "Need both operands and operators in input."

    return postfix, errors          # the return is either the expression or a message,
                                    # and a binary value indicating an error or success
