import scipy.spatial.distance as distance
from PeakType import PeakType
import numpy as np

def find_peak_opt_two(data, idx, r, threshold, c=4):
  search_path_size = r/c
  cpts = np.zeros(len(data), dtype=bool)
  window_measure = 10
  focus_val = data[idx]

  while window_measure >= threshold:
    dist_arr = distance.cdist(focus_val.reshape((1,-1)), data, 'euclidean')
    cpts[np.argwhere(dist_arr <= r)[:,1]] = True
    search_window = data[cpts]
    if len(search_window) <= 1:
      mean_val = focus_val
    else:
      mean_val = np.mean(search_window, axis=0)
    if np.linalg.norm(mean_val - focus_val) <= threshold:
      return mean_val, cpts
      break
    else:
      path_dist_arr = distance.cdist(mean_val.reshape((1,-1)), data, 'euclidean') # check if this needs to be reshaped or not/
      cpts[np.argwhere(path_dist_arr <= search_path_size)] = True
      focus_val = mean_val

def meanshift_opt_2(data, r, c):
  peak_count = -1
  peaks = np.array([])
  labels = np.zeros(len(data), dtype=int)
  labels -= 100
  threshold = 0.01

  for d_ind in (enumerate((data))):
    peak_status = PeakType.NEW_PEAK
    if labels[d_ind[0]] != -100:
      continue

    curr_peak, inside_pts = find_peak_opt_two(data, int(d_ind[0]), r, threshold, c)

    for peak_ind in range(len(peaks)):
      if (np.linalg.norm(curr_peak - peaks[peak_ind]) < float(r/2)):
        peaks[peak_ind] = curr_peak
        peak_status = PeakType.OLD_PEAK
        labels[inside_pts] = peak_ind
        break

    if peak_status is PeakType.NEW_PEAK:
      if peak_count == -1:
        peaks = np.append(peaks, curr_peak, axis=0)[np.newaxis]
        labels[inside_pts] = peak_count
        peak_count = 0
      else:
        peaks = np.append(peaks, curr_peak[np.newaxis], axis=0)
        peak_count += 1
        peaks[peak_count] = curr_peak
        labels[inside_pts] = peak_count

  return labels, peaks