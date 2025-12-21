sum1_ = 0
sum2_ = 0
cc1 = input("Please input credit card number 1:")
cc2 = input("please input credit card number 2:")
for i in range(0, 16, 2):
    x = int(cc1[i])*2
    if len(str(x)) == 1:
        sum1_ += x
    elif len(str(x)) == 2:
        xstr = str(x)
        x0 = int(xstr[0])
        x1 = int(xstr[1])
        y = x0 + x1
        sum1_ += y 
    z = int(cc2[i])*2
    if len(str(z)) == 1:
         sum2_ += z
    elif len(str(z)) == 2:
         zstr = str(z)
         z0 = int(zstr[0])
         z1 = int(zstr[1])
         q = z0 + z1
         sum2_ += q   
for j in range(1, 16, 2):
    sum1_ += int(cc1[j])
    sum2_ += int(cc2[j])
checksum1 = sum1_ % 10
checksum2 = sum2_ % 10
if checksum1 == 0:
    print("Checksum = 0")
    print(cc1 + " is a valid CC number.")
else:
    print("Checksum = " + str(checksum1))
    print(cc1 + " is an invalid CC number.")
if checksum2 == 0:
    print("Checksum = 0")
    print(cc2 + " is a valid CC number.")
else:
    print("Checksum = " + str(checksum2))
    print(cc2 + " is an invalid CC number.")