"""
Module for finding optimal multisine excitation signals for system identification.

This module provides functions to calculate frequency aliases and harmonics, generate
adjacency matrices, and create graphs for finding cliques using NetworkX.

Dependencies:
- numpy
- matplotlib
- cycler
- networkx
- schemdraw
- rwth_colors (rwth-CD-colors)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import networkx as nx
from networkx.algorithms import clique as cl
import schemdraw
import schemdraw.dsp as dsp
import schemdraw.elements as elm
import schemdraw.elements.sources as src

import rwth_colors

# Default colors for the cycler if no color is selected
rwth_colors_cycler_color = [
    rwth_colors.colors[("blue", 100)],
    rwth_colors.colors[("black", 100)],
    rwth_colors.colors[("magenta", 100)],
    rwth_colors.colors[("yellow", 100)],
    rwth_colors.colors[("green", 100)],
    rwth_colors.colors[("bordeaux", 100)],
    rwth_colors.colors[("orange", 100)],
    rwth_colors.colors[("turqoise", 100)],
    rwth_colors.colors[("darkred", 100)],
    rwth_colors.colors[("lime", 100)],
    rwth_colors.colors[("petrol", 100)],
    rwth_colors.colors[("lavender", 100)],
    rwth_colors.colors[("red", 100)],
    rwth_colors.colors[("blue", 50)],
    rwth_colors.colors[("black", 50)],
    rwth_colors.colors[("magenta", 50)],
    rwth_colors.colors[("yellow", 50)],
    rwth_colors.colors[("green", 50)],
    rwth_colors.colors[("bordeaux", 50)],
    rwth_colors.colors[("orange", 50)],
    rwth_colors.colors[("turqoise", 50)],
    rwth_colors.colors[("darkred", 50)],
    rwth_colors.colors[("lime", 50)],
    rwth_colors.colors[("petrol", 50)],
    rwth_colors.colors[("lavender", 50)],
    rwth_colors.colors[("red", 50)],
]

rwth_colors_cycler_linestyle = ["-", "--", "-."]

cc = cycler(linestyle=rwth_colors_cycler_linestyle) * cycler(
    color=rwth_colors_cycler_color
)

font_size = 10

mpl.rcParams["axes.prop_cycle"] = cc
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["axes.labelsize"] = font_size
mpl.rcParams["axes.titlesize"] = font_size
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["font.size"] = font_size
mpl.rcParams["image.cmap"] = "turbo"
mpl.rcParams["xtick.labelsize"] = font_size
mpl.rcParams["ytick.labelsize"] = font_size
mpl.rcParams["legend.fontsize"] = font_size

cm = 1 / 2.54  # Centimeters in inches

def draw_signal_path():
    """
    Draws with schemdraw a signal path with a DAC, a low-pass filter, and an ADC.

    Returns:
        fig: Figure object.
        axes: Axes object.
    """
    elm.style(elm.STYLE_IEC)

    plt.ioff()
    fig, axes = plt.subplots(1, 1)

    with schemdraw.Drawing(use_mpl=True, ax=axes, show=False) as d:
        d.config(fontsize=12, lw=1)

        sig1 = d.add(dsp.Oscillator().label('$\sin_1$','T').right())
        d.move_from(sig1.E, dx=-1,dy=-1.5)
        sig2 = d.add(dsp.Oscillator().label('$\sin_2$','T').right())
        d.move_from(sig2.E, dx=-1,dy=-1.5)
        sig3 = d.add(dsp.Oscillator().label('$\sin_{n}$','T').right().linestyle('--'))
        
        d.move_from(sig3.E, dx=0.5,dy=+1.5)
        sum_Sigma = d.add(dsp.SumSigma())

        d.add(elm.Arrow().at(sig1.E).to(sum_Sigma.NW))
        d.add(elm.Arrow().at(sig2.E).to(sum_Sigma.W))
        d.add(elm.Arrow().at(sig3.E).to(sum_Sigma.SW).linestyle('--'))

        

        d.add(elm.Line().at(sum_Sigma.E).right().length(0.5))
        
        dac = d.add(dsp.Dac().label('DAC'))
        dac_arrow = d.add(elm.Line(arrow='->').right(0.5))

        d.move_from(dac_arrow.end, dx=1,dy=-1.5)

        vcc = d.add(src.SourceControlledV().label('VCC','B').up())

        neg_bus_1 = d.add(elm.Line().at(vcc.end).right().length(2.))
        pos_bus_1 = d.add(elm.Line().at(vcc.start).right().length(2.))

        battery = d.add(src.BatteryCell().at(neg_bus_1.end).to(pos_bus_1.end).idot().dot())

        neg_bus_2 = d.add(elm.Line().at(battery.end).right().length(2.))
        pos_bus_2 = d.add(elm.Line().at(battery.start).right().length(2.))

        vm = d.add(src.MeterV().label('VM','T').at(neg_bus_2.end).to(pos_bus_2.end))

        d.move_from(vm.end, dx=1.,dy=-1.5)

        lpf_arrow = d.add(elm.Line(arrow='->').right(0.5))
        lpf = d.add(dsp.Filter(response='lp').label('LPF','T').right())

        d.add(elm.Line().right().length(.5))
        adc = d.add(dsp.Adc().label('ADC')) # ms.rwth_colors.colors[("blue", 100)]
    
    
    plt.ion()
    axes.axis('off')
    axes.axis('equal')
    plt.show()
    fig = plt.gcf()
    fig.set_size_inches(16*cm, 5*cm)

    return fig, axes

def delete_zeros(vector):
    """
    Removes all zeros in a list.

    Parameters:
        values (array-like of int): Input frequencies.

    Returns:
        ndarray of int: Array of frequencies with deleted zeros.
    """
    return np.trim_zeros(np.sort(vector))

def zero_to_nan(values):
    """
    Replaces all zeros in a list with NaN.

    Parameters:
        values (array-like of int): Input frequencies.

    Returns:
        ndarray of int: Array of frequencies with replaced zeros.
    """
    return [float('nan') if x==0 else x for x in values]



def get_alias_f(f_in, f_s):
    """
    Calculate the alias frequencies of input frequencies.

    Parameters:
        f_in (array-like of int): Input frequency or frequencies.
        f_s (int): Sampling frequency.

    Returns:
        ndarray of int: Alias frequencies.
    """
    return np.abs(f_in - np.floor(f_s / 1) * np.floor(f_in / f_s + 0.5))


def get_harmonics(f_in, harmonic_order):
    """
    Calculate harmonics of input frequencies up to a given order and adds them to a matrix.

    Parameters:
        f_in (array-like of int): Input frequencies.
        harmonic_order (int): Number of harmonics to compute.

    Returns:
        ndarray of int: Matrix of harmonics.
    """
    return np.outer(f_in, np.arange(harmonic_order) + 1)


def check_unique(matrix):
    """
    Check if all non-zero elements in the matrix are unique.

    Parameters:
        matrix (ndarray of int): Input matrix.

    Returns:
        bool: True if all non-zero elements are unique, False otherwise.
    """
    matrix_tmp = matrix.reshape(-1)
    matrix_tmp = np.sort(matrix_tmp)
    matrix_tmp = np.trim_zeros(matrix_tmp)
    return len(np.unique(matrix_tmp)) == len(matrix_tmp)


def low_pass_filter(mat_in, f_cut):
    """
    Apply a low-pass filter to the input matrix.

    Elements greater than the cutoff frequency are set to zero.

    Parameters:
        mat_in (ndarray of int): Input matrix.
        f_cut (int): Cutoff frequency.

    Returns:
        ndarray of int: Filtered matrix.
    """
    for x in np.nditer(mat_in, flags=["refs_ok"], op_flags=["readwrite"]):
        if x >= f_cut:
            x[...] = 0
    return mat_in


def is_odd(num):
    """
    Check if a number is odd.

    Parameters:
        num (int): Input number.

    Returns:
        bool: True if the number is odd, False otherwise.
    """
    num = int(num)
    return num & 0x1


def zero_to_nan(values):
    """
    Replace zeros with NaN in a list of values.

    Parameters:
        values (list or array-like): Input values.

    Returns:
        list: A new list with zeros replaced by NaN.
    """
    return [float("nan") if x == 0 else x for x in values]


def get_raw_frequencies(f_max):
    """
    Generate a range of frequencies from 0 to f_max.

    Parameters:
        f_max (float): Maximum frequency.

    Returns:
        ndarray: Array of frequencies from 0 to f_max.
    """
    f_max = int(np.floor(f_max))
    return np.arange(f_max + 1)


def get_frequencies(f_s, f_lpf, harmonics_measurable, harmonics_expected):
    """
    Compute alias frequencies after applying a low-pass filter.

    Parameters:
        f_s (int): Sampling frequency.
        f_lpf (int): Low-pass filter cutoff frequency.
        harmonics_measurable (int): Number of measurable harmonics.
        harmonics_expected (int): Number of expected harmonics.

    Returns:
        ndarray of int: Alias frequencies.
    """
    f_max = int(np.floor(f_lpf / harmonics_measurable))
    f_nyquist = f_s / 2
    f_nyquist_int = int(np.floor(f_nyquist))

    f_max = np.max([f_max,f_nyquist_int])

    return get_alias_f(
        low_pass_filter(
            get_harmonics(np.arange(f_max + 1), harmonics_expected), f_lpf
        ),
        f_s,
    )


def get_adjacency(f_s, f_lpf, harmonics_measurable, frequencies):
    """
    Generate the adjacency matrix for the graph.

    Parameters:
        f_s (int): Sampling frequency.
        f_lpf (int): Low-pass filter cutoff frequency.
        harmonics_measurable (int): Number of measurable harmonics.
        frequencies (ndarray of int): Frequency matrix.

    Returns:
        array of int: Adjacency matrix.
    """
    f_nyquist = f_s / 2
    f_nyquist_int = int(np.floor(f_nyquist))
    f_max = int(np.floor(f_lpf / harmonics_measurable))

    adjacency_mat_part = np.zeros(
        [f_nyquist_int + 1, f_nyquist_int + 1], dtype="int"
    )

    for a in range(f_nyquist_int + 1):
        for b in range(a, f_nyquist_int + 1):
            if check_unique(
                np.concatenate(
                    (frequencies[a, :], frequencies[b, :]), axis=None)
            ):
                if (
                    np.count_nonzero(frequencies[a, :] == 0) == 0
                    and np.count_nonzero(frequencies[b, :] == 0) == 0
                ):
                    adjacency_mat_part[a, b] = int(1)

    adjacency_mat_part = adjacency_mat_part + np.transpose(adjacency_mat_part)

    if is_odd(f_s):
        adjacency_mat_part_square = np.hstack(
            (adjacency_mat_part, np.flip(adjacency_mat_part, 1))
        )
        adjacency_mat_part_square = np.vstack(
            (adjacency_mat_part_square, np.flip(adjacency_mat_part_square, 0))
        )
    else:
        adjacency_mat_part_square = np.hstack(
            (adjacency_mat_part[:, :], np.flip(adjacency_mat_part[:, :-1], 1))
        )
        adjacency_mat_part_square = np.vstack(
            (
                adjacency_mat_part_square[:, :],
                np.flip(adjacency_mat_part_square[:-1, :], 0),
            )
        )

    adjacency_mat_rep = int(np.floor(f_max / f_s))

    adjacency_mat_remaining_rows = int((f_max / f_s - adjacency_mat_rep) * f_s)

    if adjacency_mat_remaining_rows == 0:
        adjacency_mat_remaining_to_delete = 0
    else:
        adjacency_mat_remaining_to_delete = f_s - adjacency_mat_remaining_rows - 1

    adjacency_mat_part_square = adjacency_mat_part_square[:-1, :-1]

    adjacency_mat = np.tile(
        adjacency_mat_part_square, [
            adjacency_mat_rep + 1, adjacency_mat_rep + 1]
    )

    if adjacency_mat_remaining_to_delete > 0:
        adjacency_mat = adjacency_mat[
            :-adjacency_mat_remaining_to_delete, :-adjacency_mat_remaining_to_delete
        ]

    adjacency_mat = np.triu(adjacency_mat, k=0)

    return adjacency_mat


def get_graph(adjacency):
    """
    Create a graph from the adjacency matrix.

    Parameters:
        adjacency (ndarray of int): Adjacency matrix.

    Returns:
        networkx.Graph: Graph object.
    """
    f_graph = nx.Graph()
    f_graph = nx.from_numpy_array(adjacency)
    edges_only = f_graph.edges
    f_graph = nx.Graph()
    f_graph.add_edges_from(edges_only)

    return f_graph


def get_cliques(
    graph, f_s, f_lpf, harmonics_measurable, harmonics_expected, search_iterations=0, cliques_per_file=100000, clique_max_length=0
):
    """
    Find cliques in the graph.

    Parameters:
        graph (networkx.Graph): Input graph.
        search_iterations (int): Number of iterations to search; 0 means search all.
        cliques_per_file (int): Number of cliques per file when saving.
        clique_max_length (int): Current maximum clique length.

    Returns:
        ndarray of int: Array of cliques found.
    """
    f_graph_cliques = cl.find_cliques(graph)

    current_clique = [[0, 0]]
    tmp_clique = [[0, 0]]

    try:
        if search_iterations == 0:
            file_name_base = (
                "cliques_fs"
                + str(f_s)
                + "_flpf"
                + str(f_lpf)
                + "_harm"
                + str(harmonics_measurable)
                + "_hare"
                + str(harmonics_expected)
            )
            file_count = 1
            while True:
                tmp_clique = next(f_graph_cliques)
                if len(tmp_clique) > clique_max_length:
                    current_clique = [tmp_clique]
                    clique_max_length = len(tmp_clique)
                elif len(tmp_clique) == clique_max_length:
                    if len(current_clique) >= cliques_per_file:
                        file_name = file_name_base + "_" + str(file_count)
                        save_cliques_as_npy(current_clique, file_name + ".npy")
                        file_count += 1
                        current_clique = [tmp_clique]
                    else:
                        current_clique.append(tmp_clique)
        else:
            for _ in range(search_iterations):
                tmp_clique = next(f_graph_cliques)
                if len(tmp_clique) > clique_max_length:
                    clique_max_length = len(tmp_clique)
                current_clique.append(sorted(tmp_clique))
    except StopIteration:
        pass

    f_graph_cliques = np.asarray(current_clique, dtype=object)
    return f_graph_cliques


def save_cliques_as_npy(cliques, filename):
    """
    Save cliques to a NumPy binary file.

    Parameters:
        cliques (list or ndarray): Cliques to save.
        filename (str): Output filename.
    """
    np.save(filename, cliques)


def load_cliques_from_npy(filename):
    """
    Load cliques from a NumPy binary file.

    Parameters:
        filename (str): Filename to load.

    Returns:
        ndarray: Loaded cliques.
    """
    return np.load(filename, allow_pickle=True)


def plot_example_undersampling():
    """
    Example plot for undersampling.
    """
    f_alias = 10
    x_lim = 7*f_alias

    pos = [1, f_alias + 3,f_alias*2+5,f_alias*3 + 7]

    x = np.linspace(0, x_lim, x_lim+1)

    y_1 = np.zeros(len(x))
    y_2 = np.zeros(len(x))
    y_3 = np.zeros(len(x))
    y_4 = np.zeros(len(x))
    y_1_a = np.zeros(len(x))
    y_2_a = np.zeros(len(x))
    y_3_a = np.zeros(len(x))
    y_4_a = np.zeros(len(x))

    y_1[pos[0]] = 1
    y_2[pos[1]] = 1
    y_3[pos[2]] = 1
    y_4[pos[3]] = 1

    y_1_a [ np.int64(get_alias_f(pos[0],f_alias*2))] = 1
    y_2_a [ np.int64(get_alias_f(pos[1],f_alias*2))] = 1
    y_3_a [ np.int64(get_alias_f(pos[2],f_alias*2))] = 1
    y_4_a [ np.int64(get_alias_f(pos[3],f_alias*2))] = 1


    fig, ax = plt.subplots(3,1,figsize=(16*cm, 11*cm), sharex=True, sharey=True)

    plt.setp(ax, xticks=[-f_alias,0, f_alias, f_alias*2,f_alias*3,f_alias*4,f_alias*5,f_alias*6, f_alias*7], xticklabels=['$-f_N$','$0$', '$f_N$', '$f_s$', '','$2f_s$','','$3f_s$',''],
            yticks=[0, 0.25, 0.5, 0.75, 1], yticklabels = ['$0$','','','',''])

    ax[0].plot([0, f_alias], [1.05, 1.05], 'C5-', lw=1,label='1st Nyquist zone')
    ax[0].plot([0, 0], [-0.05, 1.05], 'C5-', lw=1)
    ax[0].plot([0, f_alias], [-0.05, -0.05], 'C5-', lw=1)
    ax[0].plot([f_alias, f_alias], [-0.05, 1.05], 'C5-', lw=1)
    ax[0].plot([f_alias, f_alias*2], [1.05, 1.05], 'C6--', lw=1,label='2nd Nyquist zone')
    ax[0].plot([f_alias, f_alias], [-0.05, 1.05], 'C6--', lw=1)
    ax[0].plot([f_alias, f_alias*2], [-0.05, -0.05], 'C6--', lw=1)
    ax[0].plot([f_alias*2, f_alias*2], [-0.05, 1.05], 'C6--', lw=1)
    ax[0].plot([f_alias*2, f_alias*3], [1.05, 1.05], 'C7-', lw=1,label='3rd Nyquist zone')
    ax[0].plot([f_alias*2, f_alias*2], [-0.05, 1.05], 'C7-', lw=1)
    ax[0].plot([f_alias*2, f_alias*3], [-0.05, -0.05], 'C7-', lw=1)
    ax[0].plot([f_alias*3, f_alias*3], [-0.05, 1.05], 'C7-', lw=1)
    ax[0].plot([f_alias*3, f_alias*4], [1.05, 1.05], 'C8--', lw=1,label='4th Nyquist zone')
    ax[0].plot([f_alias*3, f_alias*3], [-0.05, 1.05], 'C8--', lw=1)
    ax[0].plot([f_alias*3, f_alias*4], [-0.05, -0.05], 'C8--', lw=1)
    ax[0].plot([f_alias*4, f_alias*4], [-0.05, 1.05], 'C8--', lw=1)
    ax[0].stem(x,zero_to_nan(y_1), linefmt='C0-',markerfmt='C0o', basefmt=' ')
    ax[0].stem(x,zero_to_nan(y_2), linefmt='C1-',markerfmt='C1o', basefmt=' ')
    ax[0].stem(x,zero_to_nan(y_3), linefmt='C2-',markerfmt='C2o', basefmt=' ')
    ax[0].stem(x,zero_to_nan(y_4), linefmt='C3-',markerfmt='C3o', basefmt=' ')
    ax[0].text(-.05, .99, 'a)', ha='left', va='top', transform=ax[0].transAxes)
    ax[0].legend(handlelength=0.75, loc="right")
    ax[0].grid()
    ax[0].set_ylabel('Amplitude\n in a.u.')

    ax[1].plot([-f_alias, 0], [1.05, 1.05], 'C9--', lw=1)
    ax[1].plot([-f_alias, -f_alias], [-0.05, 1.05], 'C9--', lw=1)
    ax[1].plot([-f_alias, 0], [-0.05, -0.05], 'C9--', lw=1)
    ax[1].plot([0, 0], [-0.05, 1.05], 'C9--', lw=1)
    ax[1].plot([0, f_alias], [1.05, 1.05], 'C5-', lw=1)
    ax[1].plot([0, 0], [-0.05, 1.05], 'C5-', lw=1)
    ax[1].plot([0, f_alias], [-0.05, -0.05], 'C5-', lw=1)
    ax[1].plot([f_alias, f_alias], [-0.05, 1.05], 'C5-', lw=1)
    ax[1].stem(x,zero_to_nan(y_1_a), linefmt='C0-',markerfmt='C0*', basefmt=' ')
    ax[1].stem(-x,zero_to_nan(y_2_a), linefmt='C1-',markerfmt='C1*', basefmt=' ')
    ax[1].stem(x,zero_to_nan(y_3_a), linefmt='C2-',markerfmt='C2*', basefmt=' ')
    ax[1].stem(-x,zero_to_nan(y_4_a), linefmt='C3-',markerfmt='C3*', basefmt=' ')
    ax[1].text(-.05, .99, 'b)', ha='left', va='top', transform=ax[1].transAxes)
    ax[1].grid()
    ax[1].set_ylabel('Amplitude\n in a.u.')

    ax[2].plot([0, f_alias], [1.05, 1.05], 'C5-', lw=1) 
    ax[2].plot([0, 0], [-0.05, 1.05], 'C5-', lw=1)
    ax[2].plot([0, f_alias], [-0.05, -0.05], 'C5-', lw=1)
    ax[2].plot([f_alias, f_alias], [-0.05, 1.05], 'C5-', lw=1)
    ax[2].stem(x,zero_to_nan(y_1_a), linefmt='C0-',markerfmt='C0x', basefmt=' ')
    ax[2].stem(x,zero_to_nan(y_2_a), linefmt='C1-',markerfmt='C1x', basefmt=' ')
    ax[2].stem(x,zero_to_nan(y_3_a), linefmt='C2-',markerfmt='C2x', basefmt=' ')
    ax[2].stem(x,zero_to_nan(y_4_a), linefmt='C3-',markerfmt='C3x', basefmt=' ')
    ax[2].text(-.05, .99, 'c)', ha='left', va='top', transform=ax[2].transAxes)
    ax[2].grid()
    ax[2].set_xlim([-1-f_alias,x_lim+1-f_alias+4])
    ax[2].set_ylim([-0.1,1.1])
    ax[2].set_xlabel('Frequency in Hz')
    ax[2].set_ylabel('Amplitude\n in a.u.')

    fig.tight_layout()

    return fig, ax


def plot_example_bins(x_lim = 148, f_alias = 21, f_filter = 140, f_s_plot = 42, f_excited = [1, 25, 47]):
    """
    Example plot for undersampling with harmonics.
    """
    x = np.linspace(0, x_lim, x_lim+1)

    f_1 = np.zeros(len(x))
    f_2 = np.zeros(len(x))
    f_3 = np.zeros(len(x))

    f_1_l = np.zeros(len(x))
    f_2_l = np.zeros(len(x))
    f_3_l = np.zeros(len(x))

    harm_1 = np.zeros(len(x))
    harm_2 = np.zeros(len(x))
    harm_3 = np.zeros(len(x))

    harm_1_pos = np.array(f_excited)
    for pos in harm_1_pos:
        harm_1[pos] = 1
    harm_2_pos = harm_1_pos*2
    for pos in harm_2_pos:
        harm_2[pos] = 0.5
    harm_3_pos = harm_1_pos*3
    for pos in harm_3_pos:
        harm_3[pos] = 0.25

    f_pos = 0
    f_1[harm_1_pos[f_pos]] = 1
    f_1[harm_2_pos[f_pos]] = 0.5
    f_1[harm_3_pos[f_pos]] = 0.25

    f_pos = 1
    f_2[harm_1_pos[f_pos]] = 1
    f_2[harm_2_pos[f_pos]] = 0.5
    f_2[harm_3_pos[f_pos]] = 0.25

    f_pos = 2
    f_3[harm_1_pos[f_pos]] = 1
    f_3[harm_2_pos[f_pos]] = 0.5
    f_3[harm_3_pos[f_pos]] = 0.25


    harm_1_l_pos = get_alias_f(delete_zeros(low_pass_filter(harm_1_pos, f_filter)),f_s_plot)
    harm_2_l_pos = get_alias_f(delete_zeros(low_pass_filter(harm_2_pos, f_filter)),f_s_plot)
    harm_3_l_pos = get_alias_f(delete_zeros(low_pass_filter(harm_3_pos, f_filter)),f_s_plot)


    f_pos = 0
    f_1_l[np.int64(harm_1_l_pos[f_pos])] = 1
    f_1_l[np.int64(harm_2_l_pos[f_pos])] = 0.5
    f_1_l[np.int64(harm_3_l_pos[f_pos])] = 0.25

    f_pos = 1
    f_2_l[np.int64(harm_1_l_pos[f_pos])] = 1
    f_2_l[np.int64(harm_2_l_pos[f_pos])] = 0.5
    f_2_l[np.int64(harm_3_l_pos[f_pos])] = 0.25

    f_pos = 2
    f_3_l[np.int64(harm_1_l_pos[f_pos])] = 1
    f_3_l[np.int64(harm_2_l_pos[f_pos])] = 0.5



    fig, ax = plt.subplots(1,1,figsize=(16*cm, 10*cm))


    ax.stem(x,zero_to_nan(f_1), linefmt='C0:',markerfmt='C0o', basefmt=' ',label= '1st frequency and harmonics')
    ax.stem(x,zero_to_nan(f_2) ,linefmt='C1:', markerfmt='C1o', basefmt=' ',label='2nd frequency and harmonics')
    ax.stem(x,zero_to_nan(f_3) ,linefmt='C2:', markerfmt='C2o', basefmt=' ',label='3rd frequency and harmonics')


    ax.stem(x,zero_to_nan(f_1_l), linefmt='C0-',markerfmt='C0x', basefmt=' ',label= '1st frequency undersampled')
    ax.stem(x,zero_to_nan(f_2_l) ,linefmt='C1-', markerfmt='C1x', basefmt=' ',label='2nd frequency undersampled')
    ax.stem(x,zero_to_nan(f_3_l) ,linefmt='C2-', markerfmt='C2x', basefmt=' ',label='3rd frequency undersampled')

    ax.plot([0, f_alias], [1.05, 1.05], 'C5-', lw=1,label='1st Nyquist zone')
    ax.plot([0, 0], [-0.05, 1.05], 'C5-', lw=1)
    ax.plot([0, f_alias], [-0.05, -0.05], 'C5-', lw=1)
    ax.plot([f_alias, f_alias], [-0.05, 1.05], 'C5-', lw=1)
    ax.plot([f_alias, f_alias*2], [1.05, 1.05], 'C6-', lw=1,label='2nd Nyquist zone')
    ax.plot([f_alias, f_alias], [-0.05, 1.05], 'C6-', lw=1)
    ax.plot([f_alias, f_alias*2], [-0.05, -0.05], 'C6-', lw=1)
    ax.plot([f_alias*2, f_alias*2], [-0.05, 1.05], 'C6-', lw=1)
    ax.plot([f_alias*2, f_alias*3], [1.05, 1.05], 'C7-', lw=1,label='3rd Nyquist zone')
    ax.plot([f_alias*2, f_alias*2], [-0.05, 1.05], 'C7-', lw=1)
    ax.plot([f_alias*2, f_alias*3], [-0.05, -0.05], 'C7-', lw=1)
    ax.plot([f_alias*3, f_alias*3], [-0.05, 1.05], 'C7-', lw=1)
    ax.plot([f_alias*3, f_alias*4], [1.05, 1.05], 'C8--', lw=1,label='nth Nyquist zone')
    ax.plot([f_alias*3, f_alias*3], [-0.05, 1.05], 'C8--', lw=1)
    ax.plot([f_alias*3, f_alias*4], [-0.05, -0.05], 'C8--', lw=1)


    ax.legend(handlelength=2, ncol=2,bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                    mode="expand", borderaxespad=0)

    ax.grid()
    ax.set_xlim([-2,x_lim+2])
    ax.set_ylim([-0.1,1.1])


    plt.xlabel('Frequency in Hz')
    plt.ylabel('Amplitude in a.u.')

    plt.xticks([0, f_alias, f_alias*2,f_alias*3,f_alias*4,f_alias*5,f_alias*6, f_alias*7,f_filter], ['$0$', '$f_N$', '$f_s$', '','$2f_s$','','$3f_s$','','$f_\mathrm{LPF}$'])
    plt.yticks([0, 0.25, 0.5, 0.75, 1],['$0$','','','',''])

    fig.tight_layout()
    return fig, ax