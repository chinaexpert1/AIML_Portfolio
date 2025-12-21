'''Single Qubit Operator Module by Andrew Taylor atayl136'''


# set up custom exception for operator check
class InvalidOperator(Exception):
    pass

# SingleQubitOperator class provides check and both operators
class SingleQubitOperator:
    def __init__(self, opstr):
        self.opstr = opstr
    
    # check function checks for Invalid Operators
    def check(self, opstr): 
        try:
            if opstr in 'X':
                return True
            elif opstr in 'H':
                return True                
            else:
                raise InvalidOperator
                return True
        except InvalidOperator:
            print('Invalid Operator.')
            return False

    
        
    
    
    

