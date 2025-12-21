# recursively generate a list of tuple pairs from N^2 to 1
def generate_list(N):
  if N == 1:
    return [1, 1]

  return [(N**2, N**2)] + generate_list(N-1)

print(generate_list(5))