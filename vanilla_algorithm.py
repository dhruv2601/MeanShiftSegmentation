import scipy.spatial.distance as distance
from PeakType import PeakType
import numpy as np

def findpeak(data,idx,r):
    threshold = 0.01
    focus_val = data[idx]
    window_measure = 10

    while window_measure >= threshold:
      dist_arr = distance.cdist(focus_val.reshape((1,-1)), data, 'euclidean')
      inclusion_pts_idx = np.argwhere(dist_arr < r)[:,1]
      data_inclusion = data[inclusion_pts_idx]
      if len(data_inclusion) <= 1:
        mean_val = focus_val
      else:
        mean_val = np.mean(data_inclusion, axis=0)
      
      window_measure = np.linalg.norm(mean_val - focus_val)
      focus_val = mean_val
    return mean_val, inclusion_pts_idx


def meanshift(data, r):
  peak_count = 0
  labels = np.zeros(len(data), dtype=int)
  peaks = np.array([0, 0, 0])

  for d_ind in (enumerate((data))):
    peak_status = PeakType.NEW_PEAK
    curr_peak, _ = findpeak(data, int(d_ind[0]), r)
    for peak_ind in range(len(peaks)):
      if (np.linalg.norm(curr_peak - peaks[peak_ind]) < float(r/2)):
        labels[peak_ind] = peak_ind
        peak_status = PeakType.OLD_PEAK
        break
    if peak_status is PeakType.NEW_PEAK:
      peaks = np.vstack((peaks, curr_peak))
      peak_count += 1
      labels[d_ind[0]] = peak_count
  return labels, peaks