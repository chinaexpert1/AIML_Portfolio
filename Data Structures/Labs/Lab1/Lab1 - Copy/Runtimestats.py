'''Andrew Taylor
    atayl136
    This simple module provides the runtime statistics of the program for the print statements and the output file.'''

import time
import sys



def timeandcallfunction(function, input):
    start_time = time.perf_counter_ns()
    output, errors = function(input)
    end_time = time.perf_counter_ns()
    elapsed = end_time - start_time
    return output, errors, elapsed




def sizeof(obj):
    size = sys.getsizeof(obj)
    return size









