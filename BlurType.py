'''
Enum class for two types of BLURs - Gaussian and Median
'''
from enum import Enum
class BLUR_TYPE(Enum):
  GAUSS = 1
  MEDIAN = 2