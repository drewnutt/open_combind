import pandas as pd
import numpy as np
from open_combind.utils import mp

def merge_hbonds(ifp):
    """
    Reads IFP file and merges hbond acceptors and donors.

    Setting the label to hbond for the hbond_donors and hbond_acceptors while
    changing the residue names allows for only donor+donor or acceptor+acceptor
    to be counted as overlapping, but them to be merged into the same similarity
    measure.

    Parameters
    ----------
    ifp : :class:`~pandas.DataFrame`
        IFP file read into a pandas DataFrame.

    Returns
    -------
    :class:`~pandas.DataFrame`
        IFP `DataFrame` with hbond donors and acceptors merged.
    """

    mask = ifp.label=='hbond_acceptor'
    ifp.loc[mask, 'protein_res'] = [res+'acceptor' for res in ifp.loc[mask, 'protein_res']]
    ifp.loc[mask, 'label'] = 'hbond'
    
    mask = ifp.label=='hbond_donor'
    ifp.loc[mask, 'protein_res'] = [res+'donor' for res in ifp.loc[mask, 'protein_res']]
    ifp.loc[mask, 'label'] = 'hbond'
    return ifp

def ifp_tanimoto(ifps1, ifps2, feature):
    """
    Computes the tanimoto distance between ifp1 and ifp2 for feature.

    Parameters
    ----------
    ifps1 : list of :class:`~pandas.DataFrame`
        List of IFP files read into pandas DataFrames.
    ifps2 : list of :class:`~pandas.DataFrame`
        List of IFP files read into pandas DataFrames.
    feature : str
        Feature to compute similarity for.

    Returns
    -------
    :class:`~numpy.ndarray`
        Similarity matrix.
    """
    if feature == 'hbond':
        ifps1 = [merge_hbonds(ifp) for ifp in ifps1]
        ifps2 = [merge_hbonds(ifp) for ifp in ifps2]

    ifps1 = [ifp.loc[ifp.label == feature] for ifp in ifps1]
    ifps2 = [ifp.loc[ifp.label == feature] for ifp in ifps2]

    ifps1 = [ifp.set_index('protein_res') for ifp in ifps1]
    ifps2 = [ifp.set_index('protein_res') for ifp in ifps2]

    sims = np.zeros((len(ifps1), len(ifps2)))
    for i, ifp1 in enumerate(ifps1):
        for j, ifp2 in enumerate(ifps2):
            if j > i:
                break
            total = ifp1['score'].sum() + ifp2['score'].sum()
            overlap = ifp1.join(ifp2, rsuffix='_2', how='inner')
            overlap = overlap['score']**0.5 * overlap['score_2']**0.5
            overlap = overlap.sum()

            sims[i, j] = (1 + overlap) / (2 + total - overlap)
    sims = mirror_bottom_triangle(sims)

    return sims

def ifp_tanimoto_mp(ifps1, ifps2, feature, processes):
    """
    Multiprocesing version of `ifp_tanimoto_mp`.

    Computes the tanimoto distance between ifp1 and ifp2 for feature.

    Parameters
    ----------
    ifps1 : list of :class:`~pandas.DataFrame`
        List of IFP files read into pandas DataFrames.
    ifps2 : list of :class:`~pandas.DataFrame`
        List of IFP files read into pandas DataFrames.
    feature : str
        Feature to compute similarity for.
    processes : int
        Number of processes to use.

    Returns
    -------
    :class:`~numpy.ndarray`
        Similarity matrix.

    See Also
    --------
    ifp_tanimoto: serial version of the function.
    """
    if feature == 'hbond':
        ifps1 = [merge_hbonds(ifp) for ifp in ifps1]
        ifps2 = [merge_hbonds(ifp) for ifp in ifps2]

    ifps1 = [ifp.loc[ifp.label == feature] for ifp in ifps1]
    ifps2 = [ifp.loc[ifp.label == feature] for ifp in ifps2]

    ifps1 = [ifp.set_index('protein_res') for ifp in ifps1]
    ifps2 = [ifp.set_index('protein_res') for ifp in ifps2]

    sims = np.zeros((len(ifps1), len(ifps2)))
    unfinished = [(ifp1, i, ifps2) for i, ifp1 in enumerate(ifps1)]
    results = mp(calc_sim,unfinished,processes)
    for sims_row, i in results:
        sims[i,:] = sims_row
    
    sims = mirror_bottom_triangle(sims)
    return sims

def calc_sim(ifp1,i,ifps2):
    """
    Calculates the similarity between ifp1 and each ifp2 in ifps2.

    Parameters
    ----------
    ifp1 : :class:`~pandas.DataFrame`
        IFP file read into a pandas DataFrame.
    i : int
        Index of ifp1 in ifps1.
    ifps2 : list of :class:`~pandas.DataFrame`
        List of IFP files read into pandas DataFrames.

    Returns
    -------
    :class:`~numpy.ndarray`
        1D matrix of the similarity between ifp1 and ifps2
    """

    sims = np.zeros((1,len(ifps2)))
    for j, ifp2 in enumerate(ifps2):
        if j > i:
            break
        total = ifp1['score'].sum() + ifp2['score'].sum()
        overlap = ifp1.join(ifp2, rsuffix='_2', how='inner')
        overlap = overlap['score']**0.5 * overlap['score_2']**0.5
        overlap = overlap.sum()
        sims[0,j] = (1 + overlap) / (2 + total - overlap)

    return (sims, i)

def mirror_bottom_triangle(matrix):
    """
    Mirrors the bottom triangle of a matrix to the top triangle.

    Parameters
    ----------
    matrix : :class:`~numpy.ndarray`
        Matrix with non-zero values in either top or bottom triangle and zeros elsewhere.

    Returns
    -------
    :class:`~numpy.ndarray`
        Mirrored matrix.
    """

    # This way is slow, but makes intuitive sense
    # n, m = matrix.shape
    # top_indices = np.triu_indices(n,m=m,k=1)
    # matrix[top_indices] = matrix.T[top_indices]

    # Much faster way
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))

    return matrix
