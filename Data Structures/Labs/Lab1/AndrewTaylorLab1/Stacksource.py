'''Andrew Taylor
    atayl136
This is the source code for the stacks used in the project. The data structures available in pYthon include
lists, dictionaries, sets, tuples, and strings, if you want to count strings. In  my research, arrays in Python are
 created with NumPy or Lists. To construct my stacks I chose to use Lists because strings would not hold multiple
characters as a single item. The code below sets out the functions used by my stack
to convert prefix expression to postfix format. I thought something was wrong because it ended up being very simple
but upon reflection stacks are very simple data structures so I think the code reflects the structure well.'''

# Stack Source Code

# Function to Create a Stack to be Used
# a class is not needed because we just need to create one simple list to pass in, named stack
# (we are working with one stack at a time in this program)
def createStack():
    stack = []
    return stack


# Function to Check If a Stack is Empty
# This is an important function that prevents errors from erroneous input
# it is called quite often. The program would be halted under many scenarios without it.
def isEmpty(stack):
    # this block returns True if its empty or None, False is it has contents
    if len(stack)>0:
        return False
    elif len(stack)==0:
        return True
    elif type(stack) == None:
        return True

# Push Function
# this is a simple function to push a character onto the stack using .extend()
# it pushes the character onto the stack into its own bucket
# the end of the list is the top of the stack
# it is not possible to add to a list in Python without using a function, so I chose one
def stackPush(stack, char):
    stack.extend([char])
    return stack


# Pop Function
# this is a simple function to pop a character from the top of the stack and return it using slicing
# it leans on other programming to prevent an error from popping an empty stack
# it can slice a list into NoneType, removing an empty element, so it differs from true pop in that respect
# however the isEmpty function prevents that scenario keepinf the profile of this function in line with a normal .pop()
def stackPop(stack):
    pop = stack[-1]
    stack = stack[ : -1]
    return stack, pop



