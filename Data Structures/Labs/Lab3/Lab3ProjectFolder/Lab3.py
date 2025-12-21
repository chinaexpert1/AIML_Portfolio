'''Andrew Taylor
    atayl136
    this file contains the process files and helper functions for the program.'''

from Huffman import *


# print the input list and trigger encoding or decoding

# iterate through input and encode or decode as appropriate
def call_Huffman(input, hasalpha, filename, table):
    if hasalpha:
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

        # write encoding report
        output_filename = filename.rsplit(".", 1)[0] + "_output.txt"  # one named file required for simplicity
        with open(output_filename, "w+") as sys.stdout:  # creates a text file
            print('\n\n\n\n\n')
            print("SUMMARY: MESSAGE TO ENCODE, FREQTABLE, ENCODING:")
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

    else:
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
            print("SUMMARY:")
            print('\n')
            print('List to Decode:')
            print(encoded_data)
            print('\n\n')
            print("Decoded Data:")
            for i in decoded_data:
                print(i)

