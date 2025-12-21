# import statements were kept to my files, File I/O, and time

from pathlib import Path
import sys
from Lab3 import *
from Huffman import *

# File Input section below

# This program uses a hardcoded frequency table for convenience. Any code for production would
# create a frequency table from the input. There is no need to use a named file because the user would not supply it.
# it is not considered an input file here so there is no need to read it one character at a time.


# The below line is used to get the filename argument from the command line.

# filename = sys.argv[1]
filename = "ClearText.txt"

# Create a Path object for the input file.
input_file = Path(filename)

hasalpha = False

# Open the input file in read mode and store it as a file object.
with input_file.open('r') as opened_file:
    line = ''  # Initialize an empty string to store characters one at a time.
    input = []  # Initialize an empty list to store processed expressions.

    # Code below was the code provided to read one char at a time, modified slightly to
    # join into strings for processing after being checked for validity
    # if a ValueError, it raises the exception and skips to the next line

    while True:
        char = opened_file.read(1)  # Read one character from the input file.
        if not char:  # If the character is None, it means we've reached the end of the file.
            print("End of file")
            print("\n\n")
            break

        elif char == '\t':  # Ignore tabs
            continue

        elif char == ' ':  # Ignore spaces
            continue

        elif char == '\n':  # If a new line character is found, process the line.
            print("----New Line----")
            if line != '':
                input.append(line)  # Append the processed expression to the input list.
            line = ''  # Reset the line string for the next line.

        elif char.isalpha():  # If the character is a alpha character, add it to the line. Flag that
            hasalpha = True
            line += char
            print(f"Read this char: {char}")

        elif char.isnumeric():  # If the character is a number character, add it to the line.
            line += char
            print(f"Read this char: {char}")

        else:
            continue

#########################################################################################################


# RUN PROGRAM


def main():
    call_Huffman(input, hasalpha, filename, table)

main()
