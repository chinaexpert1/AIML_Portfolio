
'''atayl136 - Module 7 Midterm Frequency Analysis \
    This program accepts a file with a string \
        in this case a ciphertext, and decrypts \
            it with a key from freq.txt'''

# read in ciphertext
reader = open("ciphertext.txt", "r")
cipherstr = reader.readline()
reader.close()

# read in frequency data and set up data structures
freqreader = open("freq.txt", "r")
freq = {}
lfreq = []
freqcount = {}
cipher = {}

# make the frequency data into a list
freqline = freqreader.readline()
while freqline != '':
    lfreq.append(freqline.split(':'))
    freqline = freqreader.readline()

freqreader.close()

# convert the list to a frequency data dictionary
for i in range(len(lfreq)):
    freq[lfreq[i][0]] = int(lfreq[i][1])

# generate a another dictionary of letter counts for each letter of ciphertext
def get_freq_counts(c, d):
    '''takes the ciphertext and outputs a dictionary \
        with the frequency count of each letter'''
    for i in range(len(c)):
        if cipherstr[i] not in d:
            d.setdefault(c[i], c.count(c[i]))
    return d
get_freq_counts(cipherstr, freqcount)

# invert frequncy data dictionary as a new dictionary
for key in freq:
    cipher.setdefault(freq[key], key)

# create a cipher to decrypt ciphertext by reassigning dict values
for key in freqcount:
    for key2 in cipher:
        if freqcount[key] == key2:
            freqcount[key] = cipher[key2]
# rename resulting dictionary  
cipher = freqcount
# create a copy of the ciphertext
hiddenmsg = cipherstr
# replace the letters in ciphertext with decrypted version according to frequency data
hiddenmsg = ''.join(cipher.get(ch, ch) for ch in hiddenmsg)
    
# output secret message
print(hiddenmsg)    
       
        
