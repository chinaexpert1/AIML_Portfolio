'''Pauli X Module by Andrew Taylor atayl136'''

import qoperators as qo
import numpy as np

class PauliX(qo.SingleQubitOperator):
    def __init__(self, opstr, qubit):
        super().__init__(opstr)
        self.qubit = qubit
        self.op = np.array([[0, 1], [1, 0]])
        self.output = 0

    # operate function applies to operator to the state    
    def operate(self, operator=None):
        if operator is None:
            operator = self.op
        self.output = np.dot(operator, self.qubit)
        return self.output
    
