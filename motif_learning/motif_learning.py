#!/usr/bin/env python
# coding: utf-8

"""
Module to learn motifs and learn how they compose with one another
"""
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from ipywidgets import interactive


class _MotifVisualisation:

    """
    Base class for plotting functions
        Attributes
    ----------
        sim_thresh
        sim_matrix
        l_motif_range
        dataset
        motif_list
        pruned_motif_list

    Methods
    -------
        matrix_plot
        plot_motif_appearances
    """

    def matrix_plot(self):
        """
        Method that plots and visualises the similarity matrix for the given dataset and sim_thresh

        Parameters
        ----------

        Returns
        -------
        self : {interatcive_plot}

        """

        def mplot(l_motif):
            sim_mtx = self.sim_matrix(l_motif) >= self.sim_thresh
            plt.figure(figsize=(8, 10))
            plt.imshow(sim_mtx, cmap="Greys")
            plt.show()
            return sim_mtx

        interactive_plot = interactive(mplot, l_motif=tuple(self.l_motif_range))
        return interactive_plot

    def plot_motif_appearances(self, pruned=True):
        """Plots the appearnces of the frequent motifs in the dataset
        Parameters
        ----------
        pruned : {bool} True means that it selects only the motifs
                that are not sub-motifs of any other motif

        Returns
        -------
        """

        def plot_single_motif(motif):
            plt.figure(figsize=(60, 3))
            plt.plot(np.arange(self.m), self.dataset, c="b", linestyle="--")
            for i, _ in enumerate(self.dataset[: -len(motif)]):
                if self.sim(self.dataset[i : i + len(motif)], motif) >= self.sim_thresh:
                    plt.scatter(
                        np.arange(i, i + len(motif)),
                        self.dataset[i : i + len(motif)],
                        c=["r"] + ["g"] * (len(motif) - 1),
                        s=45,
                    )
                    plt.hlines(y=0, xmin=0, xmax=self.m)

        if pruned: 
            self.pruned_motif_list = self.pruned_motifs()
            for motif in self.pruned_motif_list:
                plot_single_motif(motif)
        else:
            for motif in self.motif_list:
                plot_single_motif(motif)
        plt.show()


class MotifLearner(_MotifVisualisation):
    """
    A class to learn motifs in time-series data


    Attributes
    ----------
        sim_thresh
        freq_thresh
        l_motif_range
        iterator
        dataset
        frequent
        motif_list
        pruned_motif_list

    Methods
    -------
        sim
        sim_matrix
        frequent_motifs
        get_motifs
        _is_submotif
        motif_composition
        fit
        pruned_motifs
        matrix_plot
        plot_motif_appearances
    """

    def __init__(self, sim_thresh=1, freq_thresh=2, l_motif_range=[4, 20]):
        self.sim_thresh = sim_thresh
        self.freq_thresh = freq_thresh
        self.l_motif_range = l_motif_range
        self.iterator = range(*self.l_motif_range)
        self.dataset = None
        self.m = None
        self.frequent = None
        self.motif_list = None
        self.pruned_motif_list = None
        self.motif_comp_mtx = None

    def sim(self, u, v):
        """Computes the Matching similarity function between two given vectors

        Parameters
        ----------
        X : {int-list}
        Y : {int-list} of same length as X

        Returns
        -------
        similarity : {float} that equals the propotion of matching entries between X and Y.

        """
        assert len(u) == len(
            v
        ), "Length of motifs should be same to compute similarity."
        return sum(i == j for i, j in zip(u, v)) / len(u)

    def sim_matrix(self, l_motif):
        """Computes the l_motif similarity matrix on a given dataset

        Parameters
        ----------
        self.dataset : {array-like} of shape (n_samples,) containing the Training data time-series.
        l_motif : {int} specifying the motif length

        Returns
        -------
        similarity matrix: {float-matrix} of shape (n_samples, n_samples). Similarity matrix for the
                            dataset for motifs of length l_motif

        """
        sim_mtx = np.zeros((self.m - l_motif, self.m - l_motif))
        for i in range(self.m - l_motif):
            for j in range(self.m - l_motif):
                u = self.dataset[i : i + l_motif]
                v = self.dataset[j : j + l_motif]
                sim_mtx[i, j] = self.sim(u, v)
        sim_mtx = np.pad(
            sim_mtx, [(0, l_motif), (0, l_motif)], mode="constant", constant_values=0
        )
        return sim_mtx

    def frequent_motifs(self):
        """Method for finding motifs in the dataset in the given range of motif lengths

        Parameters
        ----------
        self.dataset : {array-like} of shape (n_samples,) containing the Training data time-series.
        self.sim_thresh : {float} specifies the threshold for the similarity matrix. Only if the
                        similarity is greater than or equal to sim_thresh, will the motifs be
                        considered equivalent
        self.freq_thresh: {int} specifies the minimum number of occurences for motif to considered
                        frequent
        self.l_motif_range: {int-list} of size 2. Specifies the range in which to search for motifs
                        of length.

        Returns
        -------
        frequent : {dict} that contains information of the frequent motifs, their start-point in the
                         data, and their frequencies in the form:
                    {
                    1: {start_1_motif_1: freq of 1_motif_1, start_1_motif_2: freq of 1_motif_2, ...}
                    2: {start_2_motif_1: freq of 2_motif_1, start_2_motif_2: freq of 2_motif_2, ...}
                    .
                    .
                    .
                    l_motif:{start_l_motif_1:freq of l_motif_1, start_l_motif_2:freqof l_motif_2,..}
                    .
                    .
                    .
                    }
                    if there are no motifs of length l_i, then the corresponding value for entry l
                    in dictionary is an empty dictionary

        """
        self.frequent = {}
        for i in self.iterator:
            self.frequent[i] = {}

        for l_motif in self.iterator:
            mtx = self.sim_matrix(l_motif) >= self.sim_thresh
            mtx = sparse.csr_matrix(mtx)
            sparse.csr_matrix.setdiag(mtx, 0)

            def is_lonely_point(i, j):
                if i == 0 or j == 0:
                    return not (mtx[i + 1, j + 1])
                if i == mtx.shape[0] - 1 or j == mtx.shape[0] - 1:
                    return not (mtx[i - 1, j - 1])
                return not (mtx[i + 1, j + 1] or mtx[i - 1, j - 1])

            row, _ = mtx.nonzero()
            rows = set(row)
            while rows:
                i = rows.pop()
                c_lonely = 0
                c_total = 0
                for j in mtx[i, :].nonzero()[1]:
                    if is_lonely_point(i, j):
                        c_lonely += 1
                    else:
                        c_total += 1
                if c_lonely >= self.freq_thresh:
                    c_total += c_lonely
                    self.frequent[l_motif][i] = c_total + 1
                    for point in mtx[i, :].nonzero()[1]:
                        try:
                            rows.remove(point)
                        except:
                            pass

        return self.frequent

    def get_motifs(self, plot=False):
        """Method that returns the actual motifs

        Parameters
        ----------
        plot : {bool} True creates plots of each of the frequent l-motifs
        Returns
        -------
        self : {list} of frequent motifs

        """
        self.motif_list = []
        for l_motif in self.iterator:
            if self.frequent[l_motif] != {}:
                if plot:
                    plt.figure()
                for key in self.frequent[l_motif]:
                    self.motif_list.append(list(self.dataset[key : key + l_motif]))
                    if plot:
                        plt.plot(self.dataset[key : key + l_motif])
                        plt.title(f"{l_motif}-motifs")
        return self.motif_list

    def fit(self, dataset):
        """Fit method
        Parameters
        ----------
        Returns
        -------
        self : object
        Fitted estimator.

        """
        self.dataset = dataset
        self.m = len(self.dataset)
        self.frequent = self.frequent_motifs()
        self.motif_list = self.get_motifs()

    def _is_submotif(self, big_motif, small_motif):
        """Checks whether a smaller motif is a submotif of a bigger motif

        Parameters
        ----------
        big_motif : {list} containing the bigger motif
        small_motif: {list} containing the smaller motif
        Returns
        -------
        is_submotif: {bool}
        """
        for i, _ in enumerate(big_motif):
            if np.array_equal(big_motif[i : i + len(small_motif)], small_motif):
                return True
        return False

    def motif_composition(self, as_array=True):
        """Creates a matrix that tells whether the j-th motif is a submotif of i-th motif
        This tells about the compositional structure of the motifs

        After creating a matrix of submotif relations, it creates a transitivity-free matrix
        for the Hasse diagram. This is obtained by removing edges that are there for
        transitivity. That is, wherever there are three nodes x, y,and z with edges from x
        to y and from y to z, remove the edge between x and z. This makes the compositional
        structure more readable
        """
        submotif_matrix = np.zeros((len(self.motif_list), len(self.motif_list)))
        for i, big_motif in reversed(list(enumerate(self.motif_list))):
            for j in range(i):
                small_motif = self.motif_list[j]
                if self._is_submotif(big_motif, small_motif):
                    submotif_matrix[i, j] = 1

        ### Hasse pruning of submotif_matrix
        mtx = sparse.csr_matrix(submotif_matrix)
        motif_comp_mtx = mtx.copy()
        for i, j in zip(*mtx.nonzero()):
            for k, l in zip(*mtx.nonzero()):
                if j == k:
                    motif_comp_mtx[i, l] = 0
        self.motif_comp_mtx = motif_comp_mtx.toarray()

        if as_array:
            return self.motif_comp_mtx

        return self.motif_comp_mtx

    def prune_motifs(self):
        """Creates a pruned list of motifs by removing motifs that occur as submotifs.
        Parameters
        ----------

        Returns
        -------
        pruned_motif_list: {list} if frequent motifs that are not sub motifs of other motifs
        """
        prune_idx = list(
            set(np.arange(len(self.motif_list))) - set(self.motif_comp_mtx.nonzero()[1])
        )
        self.pruned_motif_list = [self.motif_list[i] for i in prune_idx]
        return self.pruned_motif_list

    def motif_composition_analysis(self):
        """Method to analyse the motifs and how they are related to one another in terms of
        sub-motifs and super-motifs. It also prunes the set of motifs and returns the set of motifs
        that are not sub-motifs of any other motif

        Parameters
        ----------
        self

        Returns
        -------
        motif_comp_mtx : Compositional structure matrix of motifs (free of transitivtiy relations)
        pruned_motif_list : {list} of motifs that are not sub-motifs of any other motif

        """
        assert (
            self.motif_list is not None
        ), "motif_list is None\nAre you sure you ran the fit method on a dataset?"
        self.motif_composition()
        self.prune_motifs()

        return self.motif_comp_mtx, self.pruned_motif_list
