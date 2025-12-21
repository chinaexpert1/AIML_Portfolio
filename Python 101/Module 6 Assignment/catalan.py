#

'''Implementation of the Catalan number algorithm with \
    defined functions including a main function
    '''
 
###############################################################

def catalan(p):
    if p <=1:
         return 1
    
   # recursive function
    cnumber = 0
    for j in range(p):
      cnumber += catalan(j) * catalan(p-j-1)
    return cnumber
    
    
##############################################################
    
# main function

def main():
    for i in range(2, 16):
        catalan(i)
        cnumberout = catalan(i)
        print(f'Order {i} Catalan number = {cnumberout}')
        
        
main()

    
