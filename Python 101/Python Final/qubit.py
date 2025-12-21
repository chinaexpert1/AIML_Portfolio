'''Qubit Module by Andrew Taylor atayl136'''

import numpy as np

class InvalidProbabilityAmplitude(Exception):
    pass

class Qubit:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.state = np.array([[self.alpha], [self.beta]])
        self.flag = True
        self.validate_amplitudes(self.alpha, self.beta)
     
    # prob_amplitudes function was used to check experimental success
    def prob_amplitudes(self):
        alphasq = self.state[0]**2
        betasq = self.state[1]**2
        return (alphasq, betasq)
    
    # this is the required exception, and check
    def validate_amplitudes(self, alpha, beta):
        try:
            check = round(alpha**2 + beta**2) == 1
            if check == False:
                raise InvalidProbabilityAmplitude
        except InvalidProbabilityAmplitude:
            print('Invalid probability amplitude(s).')
            self.flag = False
        finally:
            return check
    
    # this is the experiment function that returns the percentage of 0's and 1's
    def experiment(self):
        self.alphasq, self.betasq = self.prob_amplitudes()
        absstate = abs(self.state)
        result = 0
        for i in range(100):
            self.exp = np.random.binomial(1, absstate)
            result += self.exp
        total = np.sum(result)
        result = result/total
        np.around(result, 2)
        return result
    
    # __str__ override so the class prints state
    def __str__(self):
        return f'{str(float(self.state[0])).replace("[", "").replace("]", "")}|0> + {str(float(self.state[1])).replace("[", "").replace("]", "")}|1>'
    
    

        