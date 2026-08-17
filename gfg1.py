# Python3 code to demonstrate working of 
# Filter Tuples by Kth element from List
# Using list comprehension

# initializing list
test_list = [("GFg", 5, 9), ("is", 4, 3), ("best", 10, 29)]

# printing original list
print("The original list is : " + str(test_list))

# initializing check_list
check_list = [4, 2, 8, 10]

# initializing K 
K = 1

# checking for presence on Kth element in list 
# one liner 
res = [sub for sub in test_list if sub[K] in check_list]

# printing result 
print("The filtered tuples : " + str(res))