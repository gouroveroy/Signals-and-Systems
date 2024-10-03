import numpy as np
import matplotlib.pyplot as plt


class ContinuousSignal:
    def __init__(self, func, INF: int):
        self.func = func
        self.INF = INF

    def shift(self, shift: int):
        return ContinuousSignal(lambda t: self.func(t - shift), self.INF)

    def add(self, other: "ContinuousSignal"):
        return ContinuousSignal(lambda t: self.func(t) + other.func(t), self.INF)

    def multiply(self, other: "ContinuousSignal"):
        return ContinuousSignal(lambda t: self.func(t) * other.func(t), self.INF)

    def multiply_constant_factor(self, scaler):
        return ContinuousSignal(lambda t: self.func(t) * scaler, self.INF)

    def plot(
        self, minheight=0, maxheight=1, y_tick_spacing=0.5, color="blue"
    ):
        t = np.linspace(-self.INF, self.INF + 0.01, 1000)
        plt.figure(figsize=(8, 3))
        plt.xticks(np.arange(-self.INF, self.INF + 1, 1))
        plt.plot(t, self.func(t), color=color)

        # Find min and max of the function
        # y_min, y_max = min(self.func(t)), max(self.func(t))

        # If minheight and maxheight are specified, use them; otherwise, use function's min and max
        # if minheight is not None and maxheight is not None:
        plt.ylim([minheight - 0.1, maxheight + 0.3])
        # else:
        #     # Add a small margin if min and max are too close or identical
        #     if y_min == y_max:
        #         y_min -= 0.1  # Set a small margin below
        #         y_max += 0.1  # Set a small margin above
        #     else:
        #         # Add 10% margin for visual clarity
        #         y_min -= 0.1 * abs(y_min)
        #         y_max += 0.1 * abs(y_max)
        #     plt.ylim([y_min, y_max])

        plt.yticks(np.arange(0, maxheight + y_tick_spacing, y_tick_spacing))

        # plt.ylim(
        #     [
        #         min(self.func(t)) - 0.1 * abs(min(self.func(t))),
        #         max(self.func(t)) + 0.1 * abs(max(self.func(t))),
        #     ]
        # )

        plt.title("Continuous Signal")
        plt.xlabel("t(Time)")
        plt.ylabel("x(t)")
        plt.grid(True)
        plt.show()

    def plot_multiple_signal(
        self,
        continuousSignals: list["ContinuousSignal"],
        title,
        supTitle,
        subplotTitles,
        rows,
        columns,
        saveTo,
        minheight=0,
        maxheight=1,
        y_tick_spacing=0.5,
        samePlot=False,
        label1="",
        label2="",
    ):
        t = np.linspace(-self.INF, self.INF + 0.01, 1000)
        reconstructed_signal = continuousSignals[len(continuousSignals) - 1]
        continuousSignals = continuousSignals[: len(continuousSignals) - 1]

        # Create a figure with multiple subplots (4 rows, 3 columns)
        fig, axs = plt.subplots(rows, columns, figsize=(10, 10))

        # Title for the entire figure
        fig.suptitle(supTitle, fontsize=16)

        # Plot the individual impulses δ/h[t-k▽] * x[t] * ▽
        k = -self.INF * 2
        row, col = 0, 0
        for continuousSignal, subplotTitle in zip(continuousSignals, subplotTitles):
            axs[row, col].set_xticks(np.arange(-self.INF, self.INF + 1, 1))
            axs[row, col].set_yticks(
                np.arange(0, maxheight + y_tick_spacing, y_tick_spacing)
            )
            if samePlot:
                axs[row, col].plot(t, continuousSignal.func(t), label=label1)
                axs[row, col].plot(t, self.func(t), color="red", label=label2)
            else:
                axs[row, col].plot(t, continuousSignal.func(t))
            axs[row, col].set_ylim([minheight, maxheight])
            axs[row, col].set_title(subplotTitle)
            axs[row, col].set_xlabel("t(Time)")
            axs[row, col].set_ylabel("x[t]")
            if samePlot:
                axs[row, col].legend()
            axs[row, col].grid(True)
            k += 1
            col += 1
            if col == columns:
                col = 0
                row += 1

        # Plot the sum of all impulse responses in the last subplot
        axs[row, col].set_xticks(np.arange(-self.INF, self.INF + 1, 1))
        axs[row, col].set_yticks(
            np.arange(0, maxheight + y_tick_spacing, y_tick_spacing)
        )
        if samePlot:
            axs[row, col].plot(t, reconstructed_signal.func(t), label=label1)
            axs[row, col].plot(t, self.func(t), color="red", label=label2)
        else:
            axs[row, col].plot(t, reconstructed_signal.func(t))
        axs[row, col].set_ylim([minheight, maxheight])
        axs[row, col].set_title(subplotTitles[len(subplotTitles) - 1])
        axs[row, col].set_xlabel("t(Time)")
        axs[row, col].set_ylabel("x[t]")
        if samePlot:
            axs[row, col].legend()
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
    signal1 = ContinuousSignal(lambda t: np.sin(t), 5)
    signal2 = ContinuousSignal(lambda t: np.cos(t), 5)

    signal1.plot()
    signal2.plot()

    # signal1.shift(2).plot()
    # signal1.add(signal2).plot()
    # signal1.multiply(signal2).plot()
    # signal1.multiply_constant_factor(2).plot()

    # signal1.shift(2).plot()
    # signal1.add(signal2).plot()
    # signal1.multiply(signal2).plot()
    # signal1.multiply_constant_factor(2).plot()
