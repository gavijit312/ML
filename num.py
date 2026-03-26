import numpy as np
# arr = np.array([1, 2, 3, 4, 5])
# print("Array:", arr)

# arr = np.array([[1, 2, 3], [4, 5, 6]])


# arr2 = np.array([[[A, B, C], [D, E, F]], [[G, H, I], 
#                                           [J, K, L]],
#                     [[M, N, O], [P, Q, R]], [[S, T, U], [V, W, X]],
#                     [[Y, Z, AA]] ])

# print(arr.ndim)
# print(arr.shape)

# rng = np.random.default_rng(seed)

# print(rng.integers(low=1, high= 101 , size =(5,2)))
#seed use for same random number every time

# print(np.random.uniform(low=0.0, high=1.0, size=(5, 2)))

rng = np.random.default_rng()

# array = np.array([1, 2, 3, 4, 5, 6])

# rng.shuffle(array)
fruits = np.array(["apple", "coconut", "banana", "pineapple"])

fruits= rng.choice(fruits)


print(fruits)