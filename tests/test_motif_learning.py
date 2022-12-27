from motif_learning import __version__
from motif_learning import MotifLearner
import numpy as np

### Loading and pre-processing data (musical notes from "Deck the halls")
notes=[]
ss= '5  43 2 1 2 3 1 23423  21 0 1   5  43 2 1 2 3 1 23423  21 0 1   2  34 2 3  45 2 345 678 7 6 5   5  43 2 1 2 3 1 66665  43 2 1   '
for elem in ss:
    if elem.isnumeric():
        notes.append(int(elem))
    else:
        notes.append(notes[-1])
notes=np.array(notes)
m = notes.size
intervals = notes[1:] - notes[:-1]
dataset=intervals

### Initialising and fitting mnotif learner to the data
motifl = MotifLearner(
        sim_thresh=1, 
        freq_thresh=2, 
        l_motif_range = [4,20])
motifl.fit(dataset)
motifl.motif_composition_analysis()

### Tests
def test_version():
    assert __version__ == '0.0.1'


def test_sim():
    u = [1,2,1,1]
    v = [1,2,1,1]
    assert motifl.sim(u,v) == 1, "Should be 1"
    

def test_sim_matrix():
    sim_mtx = motifl.sim_matrix(l_motif=5)
    assert sim_mtx.shape[0]==motifl.m, "Should be equal to the number of samples m"
    assert sim_mtx.shape[1]==motifl.m, "Should be equal to the number of samples m"


def test_get_motifs():
    ml = motifl.get_motifs()
    assert isinstance(ml, list), "Should be a list"
    assert isinstance(ml[-1], list), "Should be a list"


def test_is_submotif():
    motif_1 = [1,4,6,3,7,2]
    motif_2 = [6,3,7,2]
    assert motifl._is_submotif(motif_1, motif_2), "Should be True"
    assert not motifl._is_submotif(motif_2, motif_1), "Should be False"


def test_prune_motifs():
    ml = {tuple(i) for i in motifl.motif_list}
    pml = {tuple(i) for i in motifl.pruned_motif_list}
    assert pml.issubset(ml), "pruned_motif_list should be a subset of motif_list"

