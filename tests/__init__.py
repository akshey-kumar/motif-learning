import numpy as np
from matplotlib.pyplot import plt
from motif_learning import __version__
from motif_learning import MotifLearner

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
        
plt.figure(figsize=(30,3))
plt.plot(notes,linestyle='--', marker='.',markersize=10)

intervals = notes[1:] - notes[:-1]
plt.figure(figsize=(30,3))
plt.plot(intervals)
dataset=intervals

motifl = MotifLearner(
        sim_thresh=1, 
        freq_thresh=2, 
        l_motif_range = [4,20])
motifl.fit(dataset)
motifl.get_motifs()