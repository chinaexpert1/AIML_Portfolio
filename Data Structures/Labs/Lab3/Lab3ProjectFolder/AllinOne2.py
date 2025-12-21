

# import statements were kept to my files, File I/O, and time
from pathlib import Path
import sys


# File Input section below

# This program uses a hardcoded frequency table for convenience. Any code for production would
# create a frequency table from the input. There is no need to use a named file because the user would not supply it.
# it is not considered an input file here so there is no need to read it one character at a time.

# Open the "FreqTable.txt" file in read mode
with open('FreqTable.txt', 'r') as f:
    # Read the contents of the file
    table = f.read()
print('Frequency Table from file:')
print(table)
print('\n')


# splits the table into lines and then into a dictionary with letters as keys and frequencies as values
def parse_table(table):
    table_lines = table.split('\n')
    table_dict = {}

    for line in table_lines:
        if line != '':
            symbol, code = line.split(' - ')
            table_dict[code] = symbol

    return table_dict

table = parse_table(table)
my_dict = table
sorted_dict = {k: my_dict[k] for k in sorted(my_dict)}
table = sorted_dict
print(table)

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
    errors = 0  # start keeping track of bad characters as they are read one at a time


    # Code below was the code provided to read one char at a time, modified slightly to
    # join into strings for processing after being checked for validity
    # if a ValueError, it raises the exception and skips to the next line

    while True:
        char = opened_file.read(1)  # Read one character from the input file.
        if not char:                 # If the character is None, it means we've reached the end of the file.
            print("End of file")
            print("\n\n")
            break

        elif char == '\t':           # Ignore tabs
            continue

        elif char == ' ':           # Ignore spaces
            continue

        elif char == '\n':           # If a new line character is found, process the line.
            print("----New Line----")
            if line != '':
                input.append(line)   # Append the processed expression to the input list.
            line = ''                # Reset the line string for the next line.

        elif char.isalpha():  # If the character is a alpha character, add it to the line. Flag that
            hasalpha = True
            line += char
            print(f"Read this char: {char}")

        elif char.isnumeric():  # If the character is a number character, add it to the line.
            line += char
            print(f"Read this char: {char}")

        else:                        # If the character is not valid, skip it.
            continue

#########################################################################################################



print("SUMMARY OF INPUTS: MESSAGE TO ENCODE, FREQTABLE, AND ENCODED STRINGS TO DECODE")
print('\n')
print('Strings to Encode, ignoring spaces:')
print(input)
print('\n')
strings_list = input

# convert strings to uppercase using list comprehension
upper_strings_list = [s.upper() for s in strings_list]
input = upper_strings_list


# Open the "FreqTable.txt" file in read mode
with open('FreqTable.txt', 'r') as f:
    # Read the contents of the file
    table = f.read()
print('Frequency Table from file:')
print(table)
print('\n')

with open('Encoded.txt', 'r') as f:
    # Read the contents of the file
    codeblock = f.read()
print('Encoded Block:')
print(codeblock)
print('\n')

print('ENCODING AND DECODING PROCESS REPORT:')

def parse_table(table):
    table_lines = table.split('\n')
    table_dict = {}

    for line in table_lines:
        if line != '':
            symbol, value = line.split(' - ')
            table_dict[symbol] = value

    return table_dict

def parse_encoding(table):
    code_lines = codeblock.split('\n')

    for line in code_lines:
        if line != '':
            line = int(line)

    return code_lines

table = parse_table(table)
my_dict = table
sorted_dict = {k: my_dict[k] for k in sorted(my_dict)}
table = sorted_dict

class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        if isinstance(freq, str) and freq.isdigit():
            freq = int(freq)
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = None

    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.freq == other.freq
        return False

    def __str__(self):
        return f"{self.symbol}: {self.freq} ({self.code})"

    def is_leaf(self):
        return self.left is None and self.right is None




def insertion_sort(array):
    # Loop through array starting from second element
    for i in range(1, len(array)):
        # Store current element and initialize index variable
        key_item = array[i]
        j = i - 1

        # Compare current element with elements to its left
        while j >= 0 and array[j].freq > key_item.freq:
            # Shift elements to the right to make room
            array[j + 1] = array[j]
            j -= 1

        # Insert current element in correct position
        array[j + 1] = key_item

    # Return the sorted array
    return array




def build_huffman_tree(table):
    print('Huffman Tree Report:')
    frequencies = table
    print('Dictionary of frequencies:')
    print(frequencies)
    print('\n')
    nodes = [Node(freq, symbol) for symbol, freq in frequencies.items()]
    print('Collection of nodes without codes yet:')
    for i in nodes:
        print(i)
    print('\n')
    insertion_sort(nodes)

    while len(nodes) > 1:
        left = nodes.pop(0)
        right = nodes.pop(0)
        parent_freq = left.freq + right.freq
        combined_symbol = left.symbol + right.symbol
        parent = Node(freq=parent_freq, symbol=combined_symbol, left=left, right=right)
        nodes.append(parent)
        print( '-> Merging', left, '+', right)
        nodes.sort()


    return nodes[0]



def huffman_encoding(node, code=''):
    if node is None:
        return
    if node.left is None and node.right is None:
        node.code = code
    huffman_encoding(node.left, code + '0')
    huffman_encoding(node.right, code + '1')



def huffman_encode(data, tree):
    huffman_encoding(tree)
    print('\n')
    encoding = {}

    def fill_encoding(node):
        if node is None:
            return
        if node.left is None and node.right is None:
            encoding[node.symbol] = node.code
        fill_encoding(node.left)
        fill_encoding(node.right)

    fill_encoding(tree)

    encoded_data = ''.join([encoding[symbol] for symbol in data])
    return encoded_data

    encoded_data = ''.join([encoding[symbol] for symbol in data])
    return encoded_data


def huffman_decode(encoded_data, tree):
    root = tree
    huffman_encoding(root)


    decoded_data = ''
    current_node = root

    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        elif bit == '1':
            current_node = current_node.right

        if current_node.left is None and current_node.right is None:
            decoded_data += current_node.symbol
            current_node = root

    return decoded_data


def print_encoding(tree):
    encoding = {}
    huffman_encoding(tree)

    def fill_encoding(node):
        if node is None:
            return
        if node.left is None and node.right is None:
            encoding[node.symbol] = node.code
        fill_encoding(node.left)
        fill_encoding(node.right)

    fill_encoding(tree)
    return encoding


def print_tree(current_node):
    if current_node is None:
        return
    print(current_node.symbol, current_node.freq)
    print_tree(current_node.left)
    print_tree(current_node.right)




def post_print_tree(node, level=0):
    if node is not None:
        print_tree(node.right, level=level+1)
        if node.is_leaf():
            print(' ' * 4 * level + '->', node.symbol, node.freq)
        else:
            left_child = node.left.symbol + ':' + str(node.left.freq)
            right_child = node.right.symbol + ':' + str(node.right.freq)
            print(' ' * 4 * level + 'Internal:', left_child, '+', right_child)
            print_tree(node.left, level=level+1)





def get_height(node):
    if not node:
        return 0
    return 1 + max(get_height(node.left), get_height(node.right))


def get_level(node, level):
    if not node:
        return []
    if level == 0:
        return [node]
    left = get_level(node.left, level - 1)
    right = get_level(node.right, level - 1)
    return left + right

# Encoding example
data = "HELLOWORLD"
tree = build_huffman_tree(table)
encoded_data = huffman_encode(data, tree)
# Encoding Logic
print('\n')
print("Example Data:", data)
print("Encoded Example:", encoded_data)
print('\n')
print('Encoded ClearText:')
encoded_strings = []
for i in input:
    encoded_message_line = huffman_encode(i, tree)
    print(encoded_message_line)
    encoded_strings.append(encoded_message_line)
print('\n')
print("Huffman Tree used for encoding:")
print_tree(tree)
print('\n')
print('\n\n\n\n')
print("ENCODING AND DECODING SUMMARY:")
print('\n')
print('Encoded ClearText:')
for i in encoded_strings:
    print(i)
    print('\n')

# Decoding Logic
encoded_data = parse_encoding(codeblock)
encoded_data.pop()
print('List to Decode:')
print(encoded_data)
print('\n\n')
print("Decoded Data:")
decoded_data = []
for i in encoded_data:
    outputline = huffman_decode(i, tree)
    str(outputline)
    print(outputline)
    decoded_data.append(outputline)

output_filename = filename.rsplit(".", 1)[0] + "_output.txt"  # one named file required for simplicity
with open(output_filename, "w+") as sys.stdout:  # creates a text file
    print('\n\n\n\n\n')
    print("SUMMARY OF INPUTS: MESSAGE TO ENCODE, FREQTABLE, AND ENCODED STRINGS TO DECODE")
    print('\n')
    print('Strings to Encode, ignoring spaces:')
    print(input)
    print('\n')
    print('Frequency Table from file:')
    print(table)
    print('\n')
    print('Encoded Block:')
    print(codeblock)
    print('\n')
    print('ENCODING AND DECODING PROCESS REPORT:')
    print('\n')
    print("Example Data:", data)
    print("Encoded Example:", encoded_data)
    print('\n')

    print('\n')
    print("Huffman Tree used for encoding:")
    print_tree(tree)
    print('\n')
    print('\n\n\n\n')
    print("ENCODING AND DECODING SUMMARY:")
    print('\n')
    print('Encoded ClearText:')
    for i in encoded_strings:
        print(i)
        print('\n')
    print('List to Decode:')
    print(encoded_data)
    print('\n\n')
    print("Decoded Data:")
    for i in decoded_data:
        print(i)

#output_filename = filename.rsplit(".", 1)[0] + "_output.txt"  # one named file required for simplicity
#with open(output_filename, "w+") as f: