import numpy as np

# Stock Market Prices as a Python List
# price_list = list(map(int, input("Stock Prices: ").split()))
# n = int(input("Window size: "))
# alpha = float(input("Alpha: "))

# You may use the following input for testing purpose
price_list = [10,11,12,9,10,13,15,16,17,18]
n = 3
alpha = 0.8

# Determine the values after performing Exponential Smoothing
# The length of exsm should be = len(price_list) - n + 1
exsm = []

print("Exponential Smoothing: " + ", ".join(f"{num:.2f}" for num in exsm))
# Output should be: 11.68, 9.47, 9.82, 12.29, 14.40, 15.62, 16.64, 17.63
