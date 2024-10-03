import numpy as np
import matplotlib.pyplot as plt


class DiscreteSignal:
    def __init__(self, values: np.ndarray, INF: int):
        self.values = values
        self.INF = INF

    def set_value_at_time(self, time: int, value):
        self.values[time] = value

    def shift_signal(self, shift: int):
        new_values = self.values
        if shift > 0:
            new_values = np.concatenate(
                (np.zeros(shift), self.values[: len(self.values) - shift])
            )

        else:
            new_values = np.concatenate((self.values[-shift:], np.zeros(-shift)))
        return DiscreteSignal(new_values, self.INF)

    def add(self, other: "DiscreteSignal"):
        new_values = self.values + other.values
        return DiscreteSignal(new_values, self.INF)

    def multiply(self, other: "DiscreteSignal"):
        new_values = self.values * other.values
        return DiscreteSignal(new_values, self.INF)

    def multiply_constant_factor(self, scaler):
        new_values = self.values * scaler
        return DiscreteSignal(new_values, self.INF)

    def plot(self):
        plt.figure(figsize=(8, 3))
        plt.xticks(np.arange(-self.INF, self.INF + 1, 1))
        y_range = (-1, max(np.max(self.values), 3) + 1)
        plt.ylim(*y_range)
        plt.stem(np.arange(-self.INF, self.INF + 1, 1), self.values)
        plt.title("Discrete Signal")
        plt.xlabel("n (Time Index)")
        plt.ylabel("x[n]")
        plt.grid(True)
        plt.show()

    def plot_multiple_signal(self, DiscreteSignals: list["DiscreteSignal"], title, supTitle, subplotTitle, saveTo):
        summed_signal = DiscreteSignals[len(DiscreteSignals) - 1]
        DiscreteSignals = DiscreteSignals[: len(DiscreteSignals) - 1]

        # Create a figure with multiple subplots (4 rows, 3 columns)
        fig, axs = plt.subplots(4, 3, figsize=(10, 10))
        y_range = (-1, max(np.max(self.values), 3) + 1)

        # Title for the entire figure
        fig.suptitle(supTitle, fontsize=16)

        # Plot the individual impulses δ[n-k] * x[k]
        k = -self.INF
        row, col = 0, 0
        for DiscreteSignal in DiscreteSignals:
            axs[row, col].stem(
                np.arange(-self.INF, self.INF + 1, 1),
                DiscreteSignal.values,
                basefmt="r-",
            )
            axs[row, col].set_xticks(np.arange(-self.INF, self.INF + 1, 1))
            axs[row, col].set_ylim(*y_range)
            axs[row, col].set_title(subplotTitle+f"[n - ({k})]x[{k}]")
            axs[row, col].set_xlabel("n (Time Index)")
            axs[row, col].set_ylabel("x[n]")
            axs[row, col].grid(True)
            k += 1
            col += 1
            if col == 3:
                col = 0
                row += 1

        # Plot the sum of all impulse responses in the last subplot
        axs[row, col].stem(
            np.arange(-self.INF, self.INF + 1, 1), summed_signal.values, basefmt="r-"
        )
        axs[row, col].set_xticks(np.arange(-self.INF, self.INF + 1, 1))
        axs[row, col].set_ylim(*y_range)
        axs[row, col].set_title("Sum")
        axs[row, col].set_xlabel("n(Time Index)")
        axs[row, col].set_ylabel("x[n]")
        axs[row, col].grid(True)

        # Adjust layout to prevent overlapping of plots
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)

        # Add a caption below the figure
        fig.text(
            0.5,
            0.01,
            title,
            ha="center",
            fontsize=12,
        )

        # Save figure
        plt.savefig(saveTo)

        # Display the plot
        # plt.show()


if __name__ == "__main__":
    INF = 5
    x = DiscreteSignal(np.zeros(2 * INF + 1), INF)
    x.set_value_at_time(INF + 0, 0.5)
    x.set_value_at_time(INF + 1, 2)
    x.plot()
    # x.shift_signal(2).plot()
    # x.add(DiscreteSignal(np.ones(2 * INF + 1), INF)).plot()
    # x.multiply(DiscreteSignal(np.zeros(2 * INF + 1), INF)).plot()
    # x.multiply_constant_factor(2).plot()
