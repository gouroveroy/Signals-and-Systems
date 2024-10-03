import DiscreteSignal
import numpy as np


class LTI_Discrete:

    def __init__(self, impulse_response: "DiscreteSignal.DiscreteSignal"):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(
        self, input_signal: "DiscreteSignal.DiscreteSignal"
    ):
        INF = input_signal.INF
        unit_impulses = []
        coefficients = []
        for i in range(-INF, INF + 1):
            coefficient = input_signal.values[INF + i]
            unit_impulse = DiscreteSignal.DiscreteSignal(np.zeros(2 * INF + 1), INF)
            unit_impulse.set_value_at_time(INF, 1)
            unit_impulses.append(unit_impulse.shift_signal(i))
            coefficients.append(coefficient)

        return unit_impulses, coefficients

    def output(self, input_signal: "DiscreteSignal.DiscreteSignal"):
        INF = input_signal.INF
        constituent_impulses = []
        coefficients = []
        output_signal = DiscreteSignal.DiscreteSignal(np.zeros(2 * INF + 1), INF)
        for i in range(-INF, INF + 1):
            coefficients.append(input_signal.values[INF + i])
            response = self.impulse_response.shift_signal(i)
            constituent_impulses.append(response)
            output_signal = output_signal.add(
                response.multiply_constant_factor(input_signal.values[INF + i])
            )
        return output_signal, constituent_impulses, coefficients


if __name__ == "__main__":
    INF = 5
    impulse_response = DiscreteSignal.DiscreteSignal(np.zeros(2 * INF + 1), INF)
    impulse_response.set_value_at_time(INF + 0, 1)
    impulse_response.set_value_at_time(INF + 1, 1)
    impulse_response.set_value_at_time(INF + 2, 1)
    lti = LTI_Discrete(impulse_response)

    input_signal = DiscreteSignal.DiscreteSignal(np.zeros(2 * INF + 1), INF)
    input_signal.set_value_at_time(INF + 0, 0.5)
    input_signal.set_value_at_time(INF + 1, 2)

    input_signal.plot()

    unit_impulses, coefficients = lti.linear_combination_of_impulses(input_signal)
    for unit_impulse, coefficient in zip(unit_impulses, coefficients):
        unit_impulse.multiply_constant_factor(coefficient).plot()

    impulse_response.plot()
    output_signal, constituent_impulses, coefficients = lti.output(input_signal)

    for constituent_impulse, coefficient in zip(constituent_impulses, coefficients):
        constituent_impulse.multiply_constant_factor(coefficient).plot()

    output_signal.plot()
