import numpy as np

import DiscreteSignal
import ContinuousSignal
import LTI_Discrete
import LTI_Continuous


def main():
    # --Discrete Portion--

    INF = 5
    # --Impulse Response--
    impulse_response = DiscreteSignal.DiscreteSignal(np.zeros(2 * INF + 1), INF)
    impulse_response.set_value_at_time(INF + 0, 1)
    impulse_response.set_value_at_time(INF + 1, 1)
    impulse_response.set_value_at_time(INF + 2, 1)

    # --Discrete Signal--
    input_signal = DiscreteSignal.DiscreteSignal(np.zeros(2 * INF + 1), INF)
    input_signal.set_value_at_time(INF + 0, 0.5)
    input_signal.set_value_at_time(INF + 1, 2)

    lti = LTI_Discrete.LTI_Discrete(impulse_response)

    # --Input Portion--
    input_portion = []
    sum = DiscreteSignal.DiscreteSignal(np.zeros(2 * INF + 1), INF)
    unit_impulses, coefficients = lti.linear_combination_of_impulses(input_signal)

    for unit_impulse, coefficient in zip(unit_impulses, coefficients):
        sum = sum.add(unit_impulse.multiply_constant_factor(coefficient))
        input_portion.append(unit_impulse.multiply_constant_factor(coefficient))

    subplotTitles = []
    for k in range(-INF, INF + 1):
        subplotTitles.append(f"δ[n - ({k})]x[{k}]")

    subplotTitles.append("Sum")

    input_portion.append(sum)
    input_signal.plot(
        input_portion,
        "Figure: Returned impulses multiplied by respective coefficients",
        "Impulses multiplied by coefficients",
        subplotTitles,
        4,
        3,
        "Discrete/input.png",
    )

    # --Output Portion--
    output_portion = []
    output_signal, constituent_impulses, coefficients = lti.output(input_signal)

    for constituent_impulse, coefficient in zip(constituent_impulses, coefficients):
        output_portion.append(constituent_impulse.multiply_constant_factor(coefficient))

    subplotTitles = []
    for k in range(-INF, INF + 1):
        subplotTitles.append(f"h[n - ({k})]x[{k}]")

    subplotTitles.append("Output = Sum")

    output_portion.append(output_signal)
    output_signal.plot(
        output_portion,
        "Figure: Output",
        "Response of Input Signal",
        subplotTitles,
        4,
        3,
        "Discrete/output.png",
    )

    # --Continuous Portion--

    INF = 3
    delta = 0.5
    # --Impulse Response--
    impulse_response = ContinuousSignal.ContinuousSignal(
        lambda t: np.piecewise(t, [t < 0, t >= 0], [0, 1]), INF
    )

    # --Continuous Signal--
    input_signal = ContinuousSignal.ContinuousSignal(
        lambda t: np.piecewise(t, [t < 0, t >= 0], [0, lambda t: np.exp(-t)]), INF
    )

    lti = LTI_Continuous.LTI_Continuous(impulse_response)

    # --Input Portion--
    input_portion = []
    reconstructed_signal = ContinuousSignal.ContinuousSignal(lambda t: 0, INF)
    impulses, coefficients = lti.linear_combination_of_impulses(input_signal, delta)

    for impulse, coefficient in zip(impulses, coefficients):
        reconstructed_signal = reconstructed_signal.add(
            impulse.multiply_constant_factor(coefficient)
        )
        input_portion.append(impulse.multiply_constant_factor(coefficient))

    subplotTitles = []
    for k in range(-2 * INF, 2 * INF + 1):
        subplotTitles.append(f"δ(t - ({k}∇))x({k}∇)∇")

    subplotTitles.append("Reconstructed Signal")

    input_portion.append(reconstructed_signal)
    input_signal.plot(
        input_portion,
        "Figure: Returned impulses multiplied by their coefficients",
        "Impulses multiplied by coefficients",
        subplotTitles,
        5,
        3,
        "Continuous/input.png",
        -0.1,
        1.1,
    )

    # --Reconstructed Signal with varying Delta--
    Deltas = [0.5, 0.1, 0.05, 0.01]
    reconstructed_signals = []
    for Delta in Deltas:
        reconstructed_signal = ContinuousSignal.ContinuousSignal(lambda t: 0, INF)
        impulses, coefficients = lti.linear_combination_of_impulses(input_signal, Delta)
        for impulse, coefficient in zip(impulses, coefficients):
            reconstructed_signal = reconstructed_signal.add(
                impulse.multiply_constant_factor(coefficient)
            )
        reconstructed_signals.append(reconstructed_signal)

    subplotTitles = []
    for Delta in Deltas:
        subplotTitles.append(f"∇ = {Delta}")

    input_signal.plot(
        reconstructed_signals,
        "Figure: Reconstruction of input signal with varying delta",
        "",
        subplotTitles,
        2,
        2,
        "Continuous/input_varying_delta.png",
        -0.1,
        1.1,
        0.2,
        True,
        "Reconstructed",
        "x(t)",
    )

    # --Output Portion--
    output_portion = []
    output_signal, constituent_impulses, coefficients = lti.output_approx(
        input_signal, delta
    )

    for constituent_impulse, coefficient in zip(constituent_impulses, coefficients):
        output_portion.append(constituent_impulse.multiply_constant_factor(coefficient))

    subplotTitles = []
    for k in range(-2 * INF, 2 * INF + 1):
        subplotTitles.append(f"h(t - ({k}∇))x({k}∇)∇")

    subplotTitles.append("Output = Sum")

    output_portion.append(output_signal)
    output_signal.plot(
        output_portion,
        "Figure: Returned impulses multiplied by their coefficients",
        "Response of Impulse Signal",
        subplotTitles,
        5,
        3,
        "Continuous/output.png",
        -0.1,
        1.3,
    )

    # --Output Signal with varying Delta--
    Deltas = [0.5, 0.1, 0.05, 0.01]
    reconstructed_signals = []
    for Delta in Deltas:
        output_signal, impulses, coefficients = lti.output_approx(input_signal, Delta)
        reconstructed_signals.append(output_signal)

    subplotTitles = []
    for Delta in Deltas:
        subplotTitles.append(f"∇ = {Delta}")

    output_signal_varying_delta = ContinuousSignal.ContinuousSignal(
        lambda t: np.piecewise(t, [t < 0, t >= 0], [0, lambda t: 1 - np.exp(-t)]), INF
    )

    output_signal_varying_delta.plot(
        reconstructed_signals,
        "Figure: Approximate output signal with varying delta",
        "Approximate output as ∇ tends to 0",
        subplotTitles,
        2,
        2,
        "Continuous/output_varying_delta.png",
        -0.1,
        1.3,
        0.2,
        True,
        "y_approx(t)",
        "y(t) = (1 - e^(-t))u(t)",
    )


if __name__ == "__main__":
    main()
