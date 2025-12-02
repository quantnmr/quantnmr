import numpy as np
import matplotlib.pyplot as plt
from scale import Scale

matlab_colors = [
    (0.000, 0.447, 0.741),
    (0.850, 0.325, 0.098),
    (0.929, 0.694, 0.125),
    (0.494, 0.184, 0.556),
    (0.466, 0.674, 0.188),
    (0.000, 0.447, 0.741),
    (0.850, 0.325, 0.098),
    (0.929, 0.694, 0.125),
    (0.494, 0.184, 0.556),
]


def plot_series_1D(
    the_data=None,
    the_data_names=None,
    left_edge_ppm=None,
    right_edge_ppm=None,
    left_shift_ppm=None,
    right_shift_ppm=None,
    num_fids_to_plot=None,
    deltas=None,
    fig=None,
):

    # the_data is passed in as a list of fid series to plot
    num_series = len(the_data)
    num_points = len(the_data[0][0])

    if left_shift_ppm is None:
        left_shift_ppm = left_edge_ppm
    if right_shift_ppm is None:
        right_shift_ppm = right_edge_ppm

    if deltas is None:
        deltas = np.zeros(1)

    if num_fids_to_plot is None:
        num_fids_to_plot = len(the_data[0])

    xscale = Scale([0, num_points], [left_edge_ppm, right_edge_ppm])
    xscale_ppm = xscale.linear(np.arange(num_points))

    left = int(np.floor(xscale.linear_r(left_shift_ppm)))
    right = int(np.ceil(xscale.linear_r(right_shift_ppm)))
    s = np.s_[left:right]

    # get max and min over entire datasets
    maxy = np.NINF
    miny = np.Inf

    for r in range(num_series):
        for i in range(num_fids_to_plot):
            maxy = max(maxy, np.max(the_data[r][i, :][s]))
            miny = min(miny, np.min(the_data[r][i, :][s]))

            # if mx > maxy:
            #     maxy = mx
            # if mn < miny:
            #     miny = mn
    print(maxy, miny)
    for r in range(num_series):
        ax = fig.add_subplot(num_series, 1, r + 1)
        for i in range(num_fids_to_plot):
            color = (1 - i / (num_fids_to_plot * 1.5), 0, 0)

            if deltas.any():
                ax.plot(
                    xscale_ppm[s],
                    the_data[r][i, :][s] / maxy,
                    label=r"$\Delta$=" + f"{deltas[i]} s",
                    c=color,
                )
            else:
                ax.plot(xscale_ppm[s], the_data[r][i, :][s] / maxy, c=color)

            ax.set_xlim(xscale_ppm[s][-1], xscale_ppm[s][0])
            sminy = miny / maxy
            smaxy = maxy / maxy
            sminy = sminy - np.abs(sminy * 0.1)
            smaxy = smaxy + np.abs(smaxy * 0.1)
            ax.set_ylim(sminy, smaxy)

        ax.invert_xaxis()

        ax.set_xlabel(r"$^1$H chemical shift (ppm)")
        ax.set_ylabel(r"Intensity (Arbitrary Units)")

        if deltas.any():
            ax.legend(
                bbox_to_anchor=(1.2, 1.0),
                title=the_data_names[r],
                handlelength=1,
                fontsize=8,
                title_fontsize=8,
                labelspacing=0.2,
                loc=1,
            )

    fig.tight_layout()
    return fig
