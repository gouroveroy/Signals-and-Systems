import ContinuousSignal
import numpy as np


class LTI_Continuous:
    def __init__(self, impulse_response: "ContinuousSignal.ContinuousSignal"):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(
        self, input_signal: "ContinuousSignal.ContinuousSignal", delta: float
    ):
        # t_values = np.arange(-input_signal.INF, input_signal.INF, delta)
        # t_values = np.arange(0, input_signal.INF + delta, delta)
        t_values = np.array(
            [
                -input_signal.INF + i * delta
                for i in range(int((input_signal.INF - (-input_signal.INF)) / delta))
            ]
        )

        impulses = []
        coefficients = []

        for t in t_values:
            coefficient = input_signal.func(t) * delta
            impulse = ContinuousSignal.ContinuousSignal(
                lambda tau, t=t: (1 / delta) * ((t <= tau) & (tau <= t + delta)),
                input_signal.INF,
            )  # Impulse of width delta and height 1/delta
            impulses.append(impulse)
            coefficients.append(coefficient)

        return impulses, coefficients

    def output_approx(
        self, input_signal: "ContinuousSignal.ContinuousSignal", delta: float
    ):
        # t_values = np.arange(-input_signal.INF, input_signal.INF, delta)
        t_values = np.array(
            [
                -input_signal.INF + i * delta
                for i in range(int((input_signal.INF - (-input_signal.INF)) / delta))
            ]
        )
        constituent_impulses = []
        coefficients = []

        output_signal = ContinuousSignal.ContinuousSignal(lambda t: 0, input_signal.INF)
        for t in t_values:
            coefficients.append(input_signal.func(t) * delta)
            response = self.impulse_response.shift(t)
            constituent_impulses.append(response)
            output_signal = output_signal.add(
                response.multiply_constant_factor(input_signal.func(t) * delta)
            )
        return output_signal, constituent_impulses, coefficients


if __name__ == "__main__":
    INF = 3
    delta = 0.5
    impulse_response = ContinuousSignal.ContinuousSignal(
        lambda t: np.piecewise(t, [t < 0, t >= 0], [0, 1]), INF
    )
    lti = LTI_Continuous(impulse_response)

    input_signal = ContinuousSignal.ContinuousSignal(
        lambda t: np.piecewise(t, [t < 0, t >= 0], [0, lambda t: np.exp(-t)]), INF
    )

    input_signal.plot(0, 1, 0.2)
    reconstructed_signal = ContinuousSignal.ContinuousSignal(
        lambda t: 0, input_signal.INF
    )

    # impulses, coefficients = lti.linear_combination_of_impulses(input_signal, delta)
    # for impulse, coefficient in zip(impulses, coefficients):
    #     reconstructed_signal = reconstructed_signal.add(
    #         impulse.multiply_constant_factor(coefficient)
    #     )
    # impulse.multiply_constant_factor(coefficient).plot(0, 1)

    # reconstructed_signal.plot(0, 1, 0.5)

    output_signal, constituent_impulses, coefficients = lti.output_approx(
        input_signal, delta
    )

    for constituent_impulse, coefficient in zip(constituent_impulses, coefficients):
        constituent_impulse.multiply_constant_factor(coefficient).plot()

    output_signal.plot()
