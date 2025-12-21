'''Andrew Taylor
    atayl136
    This simple module provides the runtime statistics of the program, for the print statements and the output file.'''


# import statements, I put these here just to be safe
import time
import sys


# Timing the Process by sandwiching the call in the middle
# the whole program flows through this simple function
def timeandcallfunction(function, input):
    start_time = time.perf_counter_ns()
    output, errors = function(input)
    end_time = time.perf_counter_ns()
    elapsed = end_time - start_time
    return output, errors, elapsed


# File Size Function. I just used a built-in function to get a file size.
# I thought it would be allowed because it is not the stack source code.
# I'm not sure how to do this otherwise.
def sizeof(obj):
    size = sys.getsizeof(obj)
    return size









