'''Andrew Taylor
    atayl136
    recursive version of the postfix converter. The function takes an expression, uses an index to track the unwind
    and processes that expression into postfix, returning the postfix expression and a binary value for an error.
    '''

def postfix(expression):
    # The recursion function takes the expression and the current index as arguments,
    # processes one character at a time, and builds the postfix expression accordingly.
    # The function returns the postfix expression and the new index after processing
    # the character at the current index.
    def recursion(exp, index):
        # Base Case 1: If the index is greater than or equal to the length of the expression,
        # it means that the function has reached the end of the expression,
        # and there are no more characters left to process.
        if index >= len(exp):
            return "", index

        # Base Case 2: If the character at the current index is an alphabet (operand),
        # it means that it's a part of the final postfix expression.
        if exp[index].isalpha():
            return exp[index], index + 1

        # Check for invalid characters.
        if exp[index] not in '+-/*$' and not exp[index].isalpha():
            return f"Invalid Character: {exp[index]}", -1

        # Recursive Case: When the character at the current index is an operator,
        # the function processes the left and right operands of the operator
        # before adding the operator itself to the postfix expression.
        left, new_index = recursion(exp, index + 1)
        if new_index == -1:  # Stop further recursion if an error occurred.
            return left, -1

        right, new_index = recursion(exp, new_index)
        if new_index == -1:  # Stop further recursion if an error occurred.
            return right, -1

        # Check if either left or right operand is empty,
        # which means there are not enough operands for the current operator.
        if not left or not right:
            return "Error. Not enough operands.", -1

        return left + right + exp[index], new_index

    postfix_exp, index = recursion(expression, 0)

    # logic to return the expression or an error
    if index == len(expression):
        errors = 0
    elif index == -1:
        errors = 1
    else:
        postfix_exp = "Error. Not enough operators."
        errors = 1

    return postfix_exp, errors


