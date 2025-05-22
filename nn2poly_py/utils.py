import numpy as np

def order(x, decreasing=False):
  """
  Returns the indices that would sort an array.

  Args:
    x: The input array.
    decreasing: If True, sort in decreasing order.

  Returns:
    An array of indices that sort the input array.
  """
  if decreasing:
    return np.argsort(-np.array(x))
  else:
    return np.argsort(np.array(x))

def concat(x, y):
  """
  Concatenates two arrays.

  Args:
    x: The first array.
    y: The second array.

  Returns:
    The concatenated array.
  """
  return np.concatenate((x, y))

def prod(x):
  """
  Calculates the product of array elements.

  Args:
    x: The input array.

  Returns:
    The product of array elements.
  """
  return np.prod(x)
