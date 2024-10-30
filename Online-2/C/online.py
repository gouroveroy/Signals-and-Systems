import numpy as np
import offline


def main():
    # Input for first polynomial
    d1 = int(input("Degree of the first polynomial: "))
    poly1 = list(map(int, input("Coefficients: ").split()))

    # Input for second polynomial
    d2 = int(input("Degree of the second polynomial: "))
    poly2 = list(map(int, input("Coefficients: ").split()))

    # Multiply the polynomials using Discrete-Time Convolution
    INF = d1 + d2 + 1
    s1 = offline.DiscreteSignal(np.zeros(2 * INF + 1), INF)
    s2 = offline.DiscreteSignal(np.zeros(2 * INF + 1), INF)

    for i in range(d1 + 1):
        s1.set_value_at_time(INF + i, poly1[i])

    for i in range(d2 + 1):
        s2.set_value_at_time(INF + i, poly2[i])

    lti = offline.LTI_Discrete(s2)
    out, x, y = lti.output(s1)
    # Remove leading and trailing zeros from the output values
    coefficients = [int(value) for value in out.values]
    while coefficients and coefficients[0] == 0:
        coefficients.pop(0)
    while coefficients and coefficients[-1] == 0:
        coefficients.pop()

    # Print the result
    print(f"Degree of the Polynomial: {d1 + d2}")
    print("Coefficients: ", end="")
    for i in range(len(coefficients)):
        print(coefficients[i], end=" ")


if __name__ == "__main__":
    main()
