'''Andrew Taylor
    atayl136
This module connects the input to the postfix converter and creates output.
It is actually very simple code because that is all that was needed. This code in
contrast to recursive code uses for loops which my not be as elegant but takes up
less memory when executed.'''

# import statements for standalone testing
from datetime import datetime
from Postfixconverter import postfix


# just a couple variables to facilitate printing
inputname = 'input'
outputname = 'output'


# Function to Print the OutPut to Console
def printoutput(input, output, elapsed, inputname, outputname, filename, errors, totalerrors, inputsize, outputsize):
    print('Printing Conversion Output: \n')                     # prints the output
    for i, element in enumerate(input):                         # iterative implementation of printing
        print(f'Prefix Input # {i+1}: {element} \n')            # print statements echo input
        print(f'Printing Postfix Conversion: {output[i]} \n')   # Conversions printed
        print('\n')                                             # some spacing
    print('Done.')
    print('\n')

    print('Runtime Report:')                                    # printing runtime statistics to console
    print(f"The report took {elapsed} nanoseconds to run")
    print(f'Total lines of prefix input: {len(input)}')
    print(f'lines with bad characters {errors}')
    print(f"The size of object called {inputname} is: {inputsize} bytes")
    print(f'Total lines of output: {len(output)}')
    print(f'lines converted to postfix {len(output)-totalerrors}')
    print(f"The size of object called {outputname} is: {outputsize} bytes")
    print('Done.')
    print('\n')
    print(f'Writing Conversion Output to file "{filename}_output.txt"')
    print('Done.')                                                        # concludes output



# function to Write the Output to File
def writeoutput(input, output, elapsed, inputname, outputname, filename, errors, totalerrors, inputsize, outputsize):
    output_filename = filename.rsplit(".", 1)[0] + "_output.txt"        # one named file required for simplicity
    f= open(output_filename,"w+")                                       # creates a text file
    f.write(f"Begin Postfix Converter Log at: {datetime.now()} \n\n")   # writes to file
    for i, element in enumerate(input):                                 # iterative implementation
        f.write(f'Prefix Input # {i+1}: {element} \n')                  # writes input line
        f.write(f'Postfix Conversion: {output[i]} \n')                  # writes output
        f.write('\n')
    f.write('Done. \n')
                                                                           # confirms job

    f.write('Runtime Report: \n')                                          # writing runtime statistics to file
    f.write(f"The report took {elapsed} nanoseconds to run \n")
    f.write(f'Total lines of prefix input: {len(input)} \n')
    f.write(f'lines with bad characters {errors} \n')
    f.write(f"The size of object called {inputname} is: {inputsize} bytes \n")
    f.write(f'Total lines of output: {len(output)} \n')
    f.write(f'lines converted to postfix {len(output)-totalerrors} \n')
    f.write(f"The size of {outputname} is: {outputsize} bytes \n")
    f.write('Done. \n')                                                    # concludes output



# this function takes in the parsed and processed list of input
# as well as the named input file, passes it to the postfix converter,
# and calls the above functions to print and write the output
def process_files(input):
    output = []                                                           # builds a list of converted expressions
    postfixerrortotal = 0                                                 # tallies the errors as they come
    print(f"Begin Prefix to Postfix Converter at {datetime.now()}. \n")   # initialization message
    for i in input:
        postfixoutput, postfixerrors = postfix(i)               # Calls postfix function and receives two values
        output.append(postfixoutput)                            # puts together output
        postfixerrortotal += postfixerrors                      # Update error count
    return output, postfixerrortotal                            # returns output and error total

