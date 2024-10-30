import numpy as np
import offline

# Stock Market Prices as a Python List
price_list = list(map(int, input("Stock Prices: ").split(",")))
n = int(input("Window size: "))

# price_list = [1, 2, 3, 4, 5, 6, 7, 8]
# n = 4

# Please determine uma and wma.
INF = len(price_list) + n
s1 = offline.DiscreteSignal(np.zeros(2 * INF + 1), INF)
s2 = offline.DiscreteSignal(np.zeros(2 * INF + 1), INF)
s3 = offline.DiscreteSignal(np.zeros(2 * INF + 1), INF)

for i in range(len(price_list)):
    s1.set_value_at_time(INF + i, price_list[i])

for i in range(n):
    s2.set_value_at_time(INF + i, 1 / n)

for i in range(n):
    s3.set_value_at_time(INF + i, (n - i) / (n * (n + 1) / 2))

lti = offline.LTI_Discrete(s1)

# Unweighted Moving Averages as a Python list
uma = []
out, x, y = lti.output(s2)
uma = out.values
uma = uma[:len(uma) - n - 1]
uma = uma[len(uma) - (len(price_list) - n + 1):]

# Weighted Moving Averages as a Python list
wma = []
out, x, y = lti.output(s3)
wma = out.values
wma = wma[:len(wma) - n - 1]
wma = wma[len(wma) - (len(price_list) - n + 1):]

# Print the two moving averages
print("Unweighted Moving Averages: " + ", ".join(f"{num:.2f}" for num in uma))
print("Weighted Moving Averages:   " + ", ".join(f"{num:.2f}" for num in wma))
