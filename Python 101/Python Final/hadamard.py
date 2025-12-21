'''Hadamard Module dby Andrew Taylor atayl136'''

import qoperators as qo
import numpy as np

class Hadamard(qo.SingleQubitOperator):
    def __init__(self, opstr, qubit):
        super().__init__(opstr)
        self.qubit = qubit
        self.opm = np.array([[1, 1], [1, -1]])
        self.op = (1/(2**0.5))*self.opm
        self.output = 0
        
    # operate function applies to operator to the state  
    def operate(self, operator=None):
        if operator is None:
            operator = self.op
        self.output = np.dot(operator, self.qubit)
        return self.output

