import numpy as np
import offline


# Stock Market Prices as a Python List
price_list = list(map(int, input("Stock Prices: ").split(",")))
n = int(input("Window size: "))
alpha = float(input("Alpha: "))

# You may use the following input for testing purpose
# price_list = [10,11,12,9,10,13,15,16,17,18]
# n = 3
# alpha = 0.8

# Determine the values after performing Exponential Smoothing
# The length of exsm should be = len(price_list) - n + 1
exsm = []
INF = len(price_list) + n
s1 = offline.DiscreteSignal(np.zeros(2 * INF + 1), INF)
s2 = offline.DiscreteSignal(np.zeros(2 * INF + 1), INF)
oneminusalpha = 1.0
for i in range(n):
    s1.set_value_at_time(INF + i, alpha * oneminusalpha)
    oneminusalpha = oneminusalpha * (1 - alpha)
    
for i in range(len(price_list)):
    s2.set_value_at_time(INF + i, price_list[i])

lti = offline.LTI_Discrete(s1)
output, x, y = lti.output(s2)
exsm = output.values
m = len(price_list)
exsm = exsm[:len(exsm) - (n + 1)]
exsm = exsm[len(exsm) - (m - n + 1):]

print("Exponential Smoothing: " + ", ".join(f"{num:.2f}" for num in exsm))
# Output should be: 11.68, 9.47, 9.82, 12.29, 14.40, 15.62, 16.64, 17.63
