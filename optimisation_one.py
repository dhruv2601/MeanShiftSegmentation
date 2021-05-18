import scipy.spatial.distance as distance
from PeakType import PeakType
import numpy as np

def findpeak(data,idx,r):
    threshold = 0.01
    # value of first point    
    focus_val = data[idx]
    # initialisaton
    window_measure = 10

    # runs till the difference between current point and data >= threshold 
    while window_measure >= threshold:
      # compute eucledian distance of focus point to each point in the data.
      dist_arr = distance.cdist(focus_val.reshape((1,-1)), data, 'euclidean')
      # get points which lie in the range of radius, r
      inclusion_pts_idx = np.argwhere(dist_arr < r)[:,1]
      # data values corresponding to the points
      data_inclusion = data[inclusion_pts_idx]
      # if only one value, no mean compute required 
      if len(data_inclusion) <= 1:
        mean_val = focus_val
      # compute the new mean value to shift the window towards more dense point.
      else:
        mean_val = np.mean(data_inclusion, axis=0)
      
      # update the conditional argument      
      window_measure = np.linalg.norm(mean_val - focus_val)
      # update new peak value with the computed mean value
      focus_val = mean_val
    return mean_val, inclusion_pts_idx

def meanshift_opt_one(data, r):
  peak_count = -1
  # init empty peaks 
  peaks = np.array([])
  labels = np.zeros(len(data), dtype=int)
  # init labels with -100  
  labels -= 100

  for d_ind in (enumerate((data))):
    # init the status as new peak
    peak_status = PeakType.NEW_PEAK
    # if label has already been set for this point, skip the iteration
    if labels[d_ind[0]] != -100:
      continue

    # compute the peak for this point    
    curr_peak, inside_pts = findpeak(data, int(d_ind[0]), r)

    for peak_ind in range(len(peaks)):
      # check the constraints if the peak is new or not
      if (np.linalg.norm(curr_peak - peaks[peak_ind]) < float(r/2)):
        # update peaks array
        peaks[peak_ind] = curr_peak
        # update peak status
        peak_status = PeakType.OLD_PEAK
        for inside_idx in inside_pts:
          # update labels
          labels[inside_idx] = peak_ind
        break
    
    # if peak is indeed new peak
    if peak_status is PeakType.NEW_PEAK:
      # if this is the first peak discovered yet
      if peak_count == -1:
        peaks = np.append(peaks, curr_peak, axis=0)[np.newaxis]
        for inside_idx in inside_pts:
          # update labels
          labels[inside_idx] = peak_count
        # increase peak count by 1
        peak_count = 0
      # previous peaks existed
      else:
        # perform updates
        peaks = np.append(peaks, curr_peak[np.newaxis], axis=0)
        peak_count += 1
        peaks[peak_count] = curr_peak
        for inside_idx in inside_pts:
          labels[inside_idx] = peak_count      

  return labels, peaks