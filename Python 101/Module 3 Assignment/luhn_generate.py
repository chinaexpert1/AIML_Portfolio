invalidcc = "461480345960017"
runtotal = 0
for i in range(-1, -15, -2):
    x = int(invalidcc[i])*2
    if len(str(x)) == 1:
        runtotal += x
    elif len(str(x)) == 2:
        x0 = str(x)
        x1 = int(x0[0])
        x2 = int(x0[1])
        x3 = x1+x2
        runtotal += x3
for j in range(-2, -15, -2):
    y = int(invalidcc[j])
    runtotal += y
for k in range(-1, -15, -2):
    z = int(invalidcc[k])*2
    if len(str(z)) == 1:
        runtotal += z
    elif len(str(z)) == 2:
        z0 = str(z)
        z1 = int(z0[0])
        z2 = int(z0[1])
        z3 = z1+z2
        runtotal += z3
for h in range(-2, -15, -2):
    y = int(invalidcc[h])
    runtotal += y
checkdigit = 10-(runtotal%10)
print("The valid credit card number is: " + invalidcc + str(checkdigit) + " and the newly computed check digit is: " + str(checkdigit))
