import h5py
import numpy as np


class EventBuffer:
    def __init__(self, ev_f) -> None:
        self.f = h5py.File(ev_f, "r")
        self.x_f = self.f["x"]
        self.y_f = self.f["y"]
        self.p_f = self.f["p"]
        self.t_f = self.f["t"]

        self.fs = [self.x_f, self.y_f, self.p_f, self.t_f]

        self.n_retrieve = 100000
        self.x_cache = np.array([self.x_f[0]])
        self.y_cache = np.array([self.y_f[0]])
        self.t_cache = np.array([self.t_f[0]])
        self.p_cache = np.array([self.p_f[0]])

        self.caches = [self.x_cache, self.y_cache, self.t_cache, self.p_cache]

        self.curr_pnter = 1
    
    def update_cache(self):
        
        rx, ry, rp, rt = [e[self.curr_pnter:self.curr_pnter + self.n_retrieve] for e in self.fs]
        self.x_cache = np.concatenate([self.x_cache, rx])
        self.y_cache = np.concatenate([self.y_cache, ry])
        self.p_cache = np.concatenate([self.p_cache, rp])
        self.t_cache = np.concatenate([self.t_cache, rt])
        
        self.curr_pnter = min(len(self.t_f), self.curr_pnter + self.n_retrieve)

    def drop_cache_by_cond(self, cond):
        self.x_cache = self.x_cache[cond]
        self.y_cache = self.y_cache[cond]
        self.p_cache = self.p_cache[cond]
        self.t_cache = self.t_cache[cond]

    def retrieve_data(self, st_t, end_t):
        while self.t_cache[-1] <= end_t and (self.curr_pnter < len(self.t_f)):
            self.update_cache()
        
        ret_cond = ( st_t<= self.t_cache) & (self.t_cache <= end_t)
        ret_data = [self.t_cache[ret_cond], self.x_cache[ret_cond], self.y_cache[ret_cond], self.p_cache[ret_cond]]
        self.drop_cache_by_cond(~ret_cond)

        return ret_data

def read_timestamp_txt(txt_f):
    numbers = []
    with open(txt_f, "r") as f:
    # Read each line from the file
        for line in f:
            # Convert the line (which is a string) to an integer and append it to the numbers list
            number = float(line.strip())  # Use strip() to remove leading/trailing whitespace and newline characters
            numbers.append(number)
    
    return numbers
