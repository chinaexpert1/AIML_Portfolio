'''Andrew Taylor
    atayl136
This is the __main__ file that is used to launch the program from the command line.
It is run with a simple python command on the folder that takes a named input file as an argument.
This program reads an input file, one character at a time and checks each character for validity.
It then joins the valid characters into strings and builds a list of expressions to be processed
by calling the process_files function from a module named "Lab1".
Finally, the program outputs the processed results.'''


# import statements were kept to my files, File I/O, and time
from pathlib import Path
import time
import sys
from Lab1 import *
from Runtimestats import *



# File Input section below


# The below line is used to get the filename argument from the command line.
filename = sys.argv[1]

# Create a Path object for the input file.
input_file = Path(filename)

# Open the input file in read mode and store it as a file object.
with input_file.open('r') as opened_file:


    line = ''  # Initialize an empty string to store characters one at a time.
    input = []  # Initialize an empty list to store processed expressions.
    errors = 0  # start keeping track of bad characters as they are read one at a time


    # Code below was the code provided to read one char at a time, modified slightly to
    # join into strings for processing after being checked for validity
    # if a ValueError, it raises the exception and skips to the next line
    while True:
        char = opened_file.read(1)  # Read one character from the input file.
        try:
            if not char:                 # If the character is None, it means we've reached the end of the file.
                print("End of file")
                print("\n\n")
                break

            elif char == " ":            # Ignore spaces
                print("ignoring space")

            elif char == '\t':           # Ignore tabs
                continue

            elif char == '\n':           # If a new line character is found, process the line.
                print("----New Line----")
                if line != '':
                    input.append(line)   # Append the processed expression to the input list.
                line = ''                # Reset the line string for the next line.

            elif char.isalpha() or char in '+-*/$':  # If the character is a valid character, add it to the line.
                line += char
                print(f"Read this char: {char}")

            else:                        # If the character is not valid, raise a ValueError.
                line += char
                raise ValueError         # Raise a ValueError exception.
                # The raise statement is placed inside the try block instead of defining a custom exception because
                # it is a one-off and does not require defining a custom exception.

        except:
            print(f"Invalid character: {char}")
            print("Processing next character, but line will not be converted")
            errors += 1



# Main function: Input Processing and then Output Below


# The input list is processed, expressions are passed to the process_files function for processing.
# There is also calls for performance monitoring purposes. The program is gathered in one place.
def main(input, filename, errors):
    output, totalerrors, elapsed = timeandcallfunction(process_files, input)      # this drives the conversion
    inputsize = sizeof(input)                                                     # these two note if the conversion
    outputsize = sizeof(output)                                                   # changed the data size (it shouldn't)
    # printing and logging the output required gathering a lot of data and variables
    printoutput(input, output, elapsed, inputname, outputname, filename, errors, totalerrors, inputsize, outputsize)
    writeoutput(input, output, elapsed, inputname, outputname, filename, errors, totalerrors, inputsize, outputsize)


# Run Program
main(input, filename, errors)
# End Program











