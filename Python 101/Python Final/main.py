'''Quantum simulator by Andrew Taylor atayl136, May 10, 2022'''

import qubit as qbt
import qoperators as qo
import hadamard as hd
import paulix as px
import numpy as np

# set numpy display option
np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
df = []
# read in file data to a list
data = open("qubits.txt")
for line in data:
    df.append(line.strip().split())
data.close()


# convert alpha and beta to float
for row in range(len(df)):
    for i in range(2):
        df[row][i] = float(df[row][i])

# apply operators and experiments
for row in range(len(df)):
    qb = qbt.Qubit(df[row][0], df[row][1])
    if qb.flag == False:                   # check for amplitude validity
        continue
    else:
        print(f'Initial state: {qb}')
    for op in range(2, len(df[row])):
        opstr = str(df[row][op])
        sqo = qo.SingleQubitOperator(opstr)
        if (flag := sqo.check(opstr)) == True:   # check for operator validity
            if opstr == 'X':
                pxqb = px.PauliX(opstr, qb.state)
                qb.state = pxqb.operate()
            elif opstr == 'H':
                hdqb = hd.Hadamard(opstr, qb.state)
                qb.state = hdqb.operate()
        else:
            break
    # main output if valid only
    if flag == True:
        print(f'Final state: {qb}')
        result = qb.experiment()       
        print(f"Percentage of 0's measured: {str(result[0]).replace('[', '').replace(']', '')}")         
        print(f"Percentage of 1's measured: {str(result[1]).replace('[', '').replace(']', '')}")        
    print('\n')


