nested_tuple = ((1, 2), (3, 4), (1, 2), (5, 6))

# Flatten and get unique elements
unique_elements = set(x for tup in nested_tuple for x in tup)

print(unique_elements)  # Output: {1, 2, 3, 4, 5, 6}
