'''Andrew Taylor
    atayl136
    This file is the Huffman Encoder and Decoder, functions.'''

#########################################################################################################

from __main__ import *
from Lab3 import *


# Open the "FreqTable.txt" file in read mode
with open('FreqTable.txt', 'r') as f:
    # Read the contents of the file
    table = f.read()
print('Frequency Table from file:')
print(table)
print('\n')


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

def parse_encoding(codeblock):
    code_lines = codeblock.split('\n')

    for line in code_lines:
        if line != '':
            line = int(line)

    return code_lines


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
        print_tree(node.right, level=level +1)
        if node.is_leaf():
            print(' ' * 4 * level + '->', node.symbol, node.freq)
        else:
            left_child = node.left.symbol + ':' + str(node.left.freq)
            right_child = node.right.symbol + ':' + str(node.right.freq)
            print(' ' * 4 * level + 'Internal:', left_child, '+', right_child)
            print_tree(node.left, level=level +1)


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


