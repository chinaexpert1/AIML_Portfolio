'''Andrew Taylor
    atayl136
This is the source code for the stacks used in the project. The data structures available in pYthon include
lists, dictionaries, sets, tuples, and strings, if you want to count strings. In  my research, arrays in Python are
 created with NumPy or Lists. To construct my stacks I chose to use Lists because strings would not hold multiple
characters as a single item. The code below sets out the functions used by my stack
to convert prefix expression to postfix format.'''



# function to create a stack to be used
# a class is not needed because we just need a simple list to pass in named stack
# we are working with one stack at a time in this program
def createStack():
    stack = []
    return stack



# function to check if stack is empty
def isEmpty(stack):
    # this block returns True if its empty or None, False is it has contents
    if len(stack)>0:
        return False
    elif len(stack)==0:
        return True
    elif type(stack) == None:
        return True




# this is a simple function to push a character onto the stack using extend
# it pushes the character onto the stack into its own bucket
# the end of the list is the top of the stack
def stackPush(stack, char):
    stack.extend([char])
    return stack



# this is a simple function to pop a character from the top of the stack and return it using slicing
# it raises an exception if the stack is empty
def stackPop(stack):
    pop = stack[-1]
    stack = stack[ : -1]
    return stack, pop



