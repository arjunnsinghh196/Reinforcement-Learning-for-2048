nested_tuple = ((1, 2), (3, 4), (1, 2), (5, 6))

# Convert to a set to remove duplicate tuples
unique_tuples = set(nested_tuple)

print(unique_tuples)  # Output: {(1, 2), (3, 4), (5, 6)}
