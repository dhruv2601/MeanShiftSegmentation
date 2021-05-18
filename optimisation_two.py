import scipy.spatial.distance as distance
from PeakType import PeakType
import numpy as np

def find_peak_opt_two(data, idx, r, threshold, c=4):
  # define search path size
  search_path_size = r/c
  # cpts init - store points within the search path region
  cpts = np.zeros(len(data), dtype=bool)
  window_measure = 10
  # focus value of data
  focus_val = data[idx]

  while window_measure >= threshold:
    # compute eucledian distance of focus point to each point in the data.
    dist_arr = distance.cdist(focus_val.reshape((1,-1)), data, 'euclidean')
    # collect points which lies in the region
    cpts[np.argwhere(dist_arr <= r)[:,1]] = True
    # update search window with the selected data points
    search_window = data[cpts]    
    # if only one value, no mean compute required 
    if len(search_window) <= 1:
      mean_val = focus_val
    # compute the new mean value to shift the window towards more dense point.
    else:
      mean_val = np.mean(search_window, axis=0)
    # if the focus point is threshold near to mean point
    if np.linalg.norm(mean_val - focus_val) <= threshold:
      return mean_val, cpts
      break
    else:
      # update values
      path_dist_arr = distance.cdist(mean_val.reshape((1,-1)), data, 'euclidean')
      cpts[np.argwhere(path_dist_arr <= search_path_size)] = True
      focus_val = mean_val

def meanshift_opt_2(data, r, c):
  peak_count = -1
  # init empty peaks 
  peaks = np.array([])
  labels = np.zeros(len(data), dtype=int)
  # init labels with -100  
  labels -= 100
  threshold = 0.01

  for d_ind in (enumerate((data))):
    # init the status as new peak
    peak_status = PeakType.NEW_PEAK
    # if label has already been set for this point, skip the iteration
    if labels[d_ind[0]] != -100:
      continue

    # compute the peak for this point    
    curr_peak, inside_pts = find_peak_opt_two(data, int(d_ind[0]), r, threshold, c)

    for peak_ind in range(len(peaks)):
      # check the constraints if the peak is new or not
      if (np.linalg.norm(curr_peak - peaks[peak_ind]) < float(r/2)):
        # update peaks array
        peaks[peak_ind] = curr_peak
        # update peak status
        peak_status = PeakType.OLD_PEAK
        # update labels
        labels[inside_pts] = peak_ind
        break

    # if peak is indeed new peak
    if peak_status is PeakType.NEW_PEAK:
      # if this is the first peak discovered yet
      if peak_count == -1:
        peaks = np.append(peaks, curr_peak, axis=0)[np.newaxis]
        # update labels
        labels[inside_pts] = peak_count
        peak_count = 0
      # previous peaks existed
      else:
        # perform updates
        peaks = np.append(peaks, curr_peak[np.newaxis], axis=0)
        peak_count += 1
        peaks[peak_count] = curr_peak
        labels[inside_pts] = peak_count

  return labels, peaks