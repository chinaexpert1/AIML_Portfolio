''' Module 11 Hill Cypher by Andrew Taylor'''

import numpy as np
# inputs
mod = 26
plaintext = "ATTACKATDAWN"
plaintext_vectors = []
K1 = np.array([[19, 8, 4], [3, 12, 7]])
K2 = np.array([[7, 8], [11, 11]])
K3 = np.array([[5, 15], [4, 12]])
keys = [K1, K2, K3]

# mapping scheme to encode the plaintext
mapping_scheme = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, \
               'M': 12, 'N':13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

# function that returns the determinant
def determinant(matrix):
    '''accepts a matrix, calculates and returns its determinant'''
    def __init__(self, matrix):
        self.matrix = matrix
    det = np.linalg.det(matrix)
    return det

# Custom Exception Class
class MatrixNotInvertible(Exception):
    pass
    
# checks if a matrix is invertible
def invertible(matrix):
    '''tests if a matrix is invertible, raises an exception if it is not.'''
    def __init__(self, matrix):
        self.matrix = matrix
    # custom exception checks
    if matrix.shape[0] != matrix.shape[1]:
        raise MatrixNotInvertible("The matrix is not square.")
        return False
    det_to_check = determinant(matrix)
    if det_to_check == 0:
       raise MatrixNotInvertible("The determinant = 0.")
       return True
    else:        
        print("The matrix is invertible.")
        return True  

# modular multiplicative inverse function
def mod_inverse(n, m):
    '''returns mmi or exception if doesnt exist'''
    for x in range(1,m):
        if((n%m)*(x%m) % m==1):
            return x
    raise Exception('The modular inverse does not exist.')

def encrypt(KEY, plaintext):
    # check if key is invertible
    invertible(KEY)
    # set up plaintext vectors
    for i in range(0, len(plaintext), 2):
        plaintext_vectors.append([plaintext[i], plaintext[i+1]])   
    # set up a copy of plaintext vectors to encode
    pt_column_vectors = plaintext_vectors
    # Convert plaintext vectors to encoded vectors
    for i in range(len(plaintext_vectors)):
        for j in range(2):
            for key in mapping_scheme:
                if plaintext_vectors[i][j] == key:
                    pt_column_vectors[i][j] = mapping_scheme[key]               
    # add encoded vectors to an array
    encoded_text_array = []
    for i in range(len(pt_column_vectors)):
        encoded_text_array.append(np.array(pt_column_vectors[i]))   
    # encrypt encoded text array    
    ciphertext_array = []
    for vector in range(len(encoded_text_array)):
        cvalue = np.dot(KEY, encoded_text_array[vector])
        ciphertext_array.append(cvalue % mod)
    # convert encrypted array to a list to work with
    ciphertext = []
    for column in range(len(ciphertext_array)):
        ciphertext.append(ciphertext_array[column].tolist()) 
    # translate that list according to mapping scheme
    for i in range(len(ciphertext)):
        for j in range(2):
            for key in mapping_scheme:
                if ciphertext[i][j] == mapping_scheme[key]:
                    ciphertext[i][j] = key              
    # construct ciphertext from mapped list
    cptext = ""
    for i in range(len(ciphertext)):
        for j in range(2):
            cptext += ciphertext[i][j]   
    return ciphertext_array, cptext

def decrypt(ekey, encrypted_array):
  
   # preparing decrypt key
    ekey_inv = np.linalg.inv(np.array(ekey))
    det = determinant(ekey)
    det = round(det)
    ekey_inv2 = ekey_inv * det
    ekey_inv_mod = ekey_inv2 % mod
    mmidet = mod_inverse(det, mod)
    ekeyfinal = (ekey_inv_mod * mmidet)
    ekeyfinal = ekeyfinal.astype(int)

    # take the encrypted_array and decrypt it with the key
    decrypted_array = []
    for vector in range(len(encrypted_array)):
        dvalue = np.dot(ekeyfinal, encrypted_array[vector])
        decrypted_array.append(dvalue % mod)
    
    # take the decrypted array and translate it back to plaintext
    dlist = []
    decryptedtext = ""
    for i in range(len(decrypted_array)):
        for j in range(2):
            for key in mapping_scheme:
                if decrypted_array[i][j] == mapping_scheme[key]:
                    dlist.append(key)                 
    decryptedtext = "".join(map(str, dlist)) 
    return decrypted_array, decryptedtext

def main():
    for k in range(3):
        try:
            array_encrypted, ciphertext = encrypt(keys[k], plaintext)
            print("Ciphertext: "+ ciphertext)
            print("Ciphertext column vectors: ", array_encrypted, "\n")
            array_out, decrypted_text = decrypt(keys[k], array_encrypted)
            print("Plaintext: " + decrypted_text)
            print("Plaintext column vectors: ", array_out, "\n")
        except MatrixNotInvertible:
            if keys[k].shape[0] != keys[k].shape[1]:
                print("The matrix is not square. \n")
            elif determinant(keys[k]) == 0:
                print("The determinant = 0. \n")
    
if __name__ == "__main__":
    main()





