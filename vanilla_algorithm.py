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


def meanshift(data, r):
  peak_count = 0
  # init labels with zeros
  labels = np.zeros(len(data), dtype=int)
  # init peaks with single value
  peaks = np.array([0, 0, 0])

  for d_ind in (enumerate((data))):
    # init the status as new peak
    peak_status = PeakType.NEW_PEAK
    # compute the peak for this point
    curr_peak, _ = findpeak(data, int(d_ind[0]), r)
    for peak_ind in range(len(peaks)):
      # check the constraints if the peak is new or not
      if (np.linalg.norm(curr_peak - peaks[peak_ind]) < float(r/2)):
        # update labels
        labels[peak_ind] = peak_ind
        # update peak status
        peak_status = PeakType.OLD_PEAK
        break
    # new peak is found
    if peak_status is PeakType.NEW_PEAK:
      # add new peak
      peaks = np.vstack((peaks, curr_peak))
      # update peak count
      peak_count += 1
      # update label
      labels[d_ind[0]] = peak_count
  return labels, peaks