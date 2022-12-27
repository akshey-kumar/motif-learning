from motif_learning import __version__
from motif_learning import MotifLearner
import numpy as np

def test_version():
    assert __version__ == '0.0.1'

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
motifl = MotifLearner(
        sim_thresh=1, 
        freq_thresh=2, 
        l_motif_range = [4,20])
motifl.fit(dataset)
motifl.get_motifs()

def test_get_motifs():
    ml = motifl.get_motifs()
    print(ml)
    assert isinstance(ml, list)
    assert isinstance(ml[-1], list)
