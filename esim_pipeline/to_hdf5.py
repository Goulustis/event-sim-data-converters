import h5py
import numpy as np
import os.path as osp

# EventCD = np.dtype({'names':['x','y','p','t'],
#                     'formats':['<u2','<u2','i2', '<i8'],
#                     'offsets':[0,2,4,8],
#                     'itemsize':16})
EventCD = np.dtype({'names':['t', 'x','y','p'],
                    'formats':[ 'u8', '<u2','<u2','<i1']})

def main():
    event_f = "/home/hunter/projects/esim_pipeline/generated_data/unif_carpet_2048/events.txt"
    targ_f = osp.join(osp.dirname(event_f), "events.h5")

    print("loading txt events")
    events = np.loadtxt(event_f, dtype=EventCD, delimiter=" ", skiprows=1)

    print("forming hdf5 events")
    with h5py.File(targ_f, "w") as hf:
        for e in "xyt":
            data = events[e]
            hf.create_dataset(e, data=data, shape=data.shape, dtype=EventCD[e])
        
        ps = events["p"]
        ps[ps == 0] = -1
        hf.create_dataset("p", data=ps, shape=ps.shape, dtype=EventCD["p"])


if __name__ == "__main__":
    main()