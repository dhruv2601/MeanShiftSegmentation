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

def meanshift_opt_one(data, r):
  peak_count = -1
  peaks = np.array([])
  labels = np.zeros(len(data), dtype=int)
  labels -= 100

  for d_ind in (enumerate((data))):
    peak_status = PeakType.NEW_PEAK
    if labels[d_ind[0]] != -100:
      continue

    curr_peak, inside_pts = findpeak(data, int(d_ind[0]), r)

    for peak_ind in range(len(peaks)):
      if (np.linalg.norm(curr_peak - peaks[peak_ind]) < float(r/2)):
        peaks[peak_ind] = curr_peak
        peak_status = PeakType.OLD_PEAK
        for inside_idx in inside_pts:
          labels[inside_idx] = peak_ind
        break

    if peak_status is PeakType.NEW_PEAK:
      if peak_count == -1:
        peaks = np.append(peaks, curr_peak, axis=0)[np.newaxis]
        for inside_idx in inside_pts:
          labels[inside_idx] = peak_count
        peak_count = 0
      else:
        peaks = np.append(peaks, curr_peak[np.newaxis], axis=0)
        peak_count += 1
        peaks[peak_count] = curr_peak
        for inside_idx in inside_pts:
          labels[inside_idx] = peak_count      

  return labels, peaks