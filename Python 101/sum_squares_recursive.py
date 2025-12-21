def sum_squares(N):
  if N == 1:
    print("Returning 1^2 = 1")
    return 1

  print("Computing: " + str(N) + "^2, but first...")

  return N**2 + sum_squares(N-1)

# compute 5^2 + 4^2 + 3^2 + 2^2 + 1^2
print(sum_squares(5))