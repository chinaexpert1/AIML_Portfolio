

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

# call this function to build the dictionary of frequencies
table = parse_table(table)
my_dict = table
sorted_dict = {k: my_dict[k] for k in sorted(my_dict)}
table = sorted_dict
print(table)

# The below line is used to get the filename argument from the command line.
# I commented out the ability to read from named files because it was inconvenient in testing
# filename = sys.argv[1]
filename = "ClearText.txt"


# Create a Path object for the input file.
input_file = Path(filename)
# this is a flag that comes in to use for named files. I allows ClearText and Encoded files to use the same program
hasalpha = False

# Open the input file in read mode and store it as a file object.
with input_file.open('r') as opened_file:

    line = ''  # Initialize an empty string to store characters one at a time.
    input = []  # Initialize an empty list to store processed expressions.


    # Code below was the code provided to read one char at a time, modified slightly to
    # join into strings for processing after being checked for validity

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


# I start my report here when the inputs are prepared. It made it easier for me to keep track of
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


# For the encoded strings I didn't read one character at a time because it is assumed not to have errors
# In my view this is not data as defined, this is provided as is by the user.
# if it had errors the message wouldn't come out right. I didn't know if that could be checked, I think it can't.
# This Lab didn't mention error checking in the instructions so I didn't implement it.
with open('Encoded.txt', 'r') as f:
    # Read the contents of the file
    codeblock = f.read()
print('Encoded Block:')
print(codeblock)           # prints out numbers to decode as a confirmation
print('\n')

# title the report
print('ENCODING AND DECODING PROCESS REPORT:')

# this is simply to convert the strings into ints so it can be decoded
def parse_table(table):
    table_lines = table.split('\n')
    table_dict = {}

    for line in table_lines:
        if line != '':
            symbol, value = line.split(' - ')
            table_dict[symbol] = value

    return table_dict

# this is simply to convert the strings into ints so it can be decoded
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

# the essentials of the tree start here with a Node class. The tree is built of these nodes which are nested
# in the left and right pointers.

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



# here is the insertion sort I implemented. It seems to work fine for a list of nodes,
# which was called once to order the nodes for merging

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


# this function does the heavy lifting in the program, it builds the Huffman Tree.
# it starts by creating a list of nodes from the frequency table, and orders them and merges them.

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

        # In theory thie merging should work without a sort. Scott Almes also didn't know why this was necessary when the nodes start
        # ordered. my insertion sort works I confirmed it in the print statements but I had to add this for some reason. It is a built-in function
        # This is not a library function.
        nodes.sort()


    return nodes[0]




# this function adds codes on to the leaves of the tree after the Huffman Tree is built
def huffman_encoding(node, code=''):
    if node is None:
        return
    if node.left is None and node.right is None:
        node.code = code
    huffman_encoding(node.left, code + '0')
    huffman_encoding(node.right, code + '1')



# this is the function to encode a cleartext line after it has been read into the input list
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

# this is the function to decode a line of encoded text according to the Huffman Tree after it is built
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

# this function works with the other function to add codes
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

# this is the function to print the tree using a preorder traversal
def print_tree(current_node):
    if current_node is None:
        return
    print(current_node.symbol, current_node.freq)
    print_tree(current_node.left)
    print_tree(current_node.right)



# this was an optional formatting function to print the tree with indentation and internal nodes marked
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




# this gets the height of the tree
def get_height(node):
    if not node:
        return 0
    return 1 + max(get_height(node.left), get_height(node.right))

# this gets the level of a node
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

# this writes the output to a file named output.txt
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

# end program
# I didnt make this modular because that gave me trouble, but I don't think it needs it because its a page and a half.
# I also didn't write a main() function. My idea is that this is written for a task, not a repeated problem
# because the frequency table doesn't change. In the end I could have made this program
# better but I'm just happy that it works. Thank you.