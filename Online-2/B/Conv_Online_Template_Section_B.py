import numpy as np

# Stock Market Prices as a Python List
price_list = list(map(int, input("Stock Prices: ").split()))
n = int(input("Window size: "))

# price_list = [1, 2, 3, 4, 5, 6, 7, 8]
# n = 4

# Please determine uma and wma.

# Unweighted Moving Averages as a Python list
uma = []

# Weighted Moving Averages as a Python list
wma = []

# Print the two moving averages
print("Unweighted Moving Averages: " + ", ".join(f"{num:.2f}" for num in uma))
print("Weighted Moving Averages:   " + ", ".join(f"{num:.2f}" for num in wma))
