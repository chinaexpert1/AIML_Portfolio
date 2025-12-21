

# recursive relationship via col and row
def number(col,row):
    if(col == 0) or (col == row):
        return 1
    else:
        return number(col-1,row-1) + number(col,row-1)

# iterate over for recursion for n rows
def pascal(n):

    for r in range(n):
        for c in range(r+1):
            print(number(c,r), end=' ')   
        print('\n')
      

def main():
    pascal(10)
    
main()