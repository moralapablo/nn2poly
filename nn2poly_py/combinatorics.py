import itertools
import collections # Keep for potential future use e.g. Counter
import numpy as np # Keep for potential future use
from .utils import order # Relative import for package structure

def combinations_with_repetition(n, k):
  """
  Generates combinations with repetition.

  Args:
    n: The number of items to choose from (from 0 to n-1).
    k: The number of items to choose.

  Returns:
    A list of tuples, where each tuple is a combination.
    Example: combinations_with_repetition(2, 2) -> [(0,0), (0,1), (1,1)]
  """
  if k == 0:
    return [()] # Standard: one way to choose 0 items is an empty tuple
  return list(itertools.combinations_with_replacement(range(n), k))

class MultisetPartitions:
  """
  Generates partitions of a multiset into k non-empty parts.

  This class implements a recursive algorithm to find all unique ways to
  partition a given multiset into exactly k non-empty sub-multisets.
  The order of elements within sub-multisets does not matter (they are sorted).
  The order of sub-multisets within a partition does not matter (list of parts is sorted).
  """

  def __init__(self, multiset, k):
    """
    Initializes the MultisetPartitions generator.

    Args:
      multiset: A list or tuple representing the multiset. Elements should be comparable.
                Example: (0, 0, 1, 2, 2, 2) for the multiset {0,0,1,2,2,2}.
      k: The desired number of non-empty parts in each partition.
    """
    if not isinstance(multiset, (list, tuple)):
      raise TypeError("multiset must be a list or tuple.")
    if not isinstance(k, int): # k can be 0
      raise ValueError("k must be an integer.")
    if k < 0:
      raise ValueError("k cannot be negative.")

    self.multiset = sorted(list(multiset))
    self.k = k
    self.n = len(self.multiset)
    self._partitions = []

    # Handle edge cases upfront
    if self.k == 0:
      if self.n == 0:
        # Partition of an empty set into 0 parts is one partition: the empty partition (list of 0 parts).
        # Some define this as [[]] (list containing one empty list, representing the empty collection of parts).
        # C++ returns List::create(List()) which is like [[]]
        # Let's use [[]] to signify a list containing one "configuration" which is an empty list of parts.
        # If the expectation is just an empty list of parts, it would be [].
        # Given C++ `List::create(R_NilValue)` (empty list) for (X,0) where X not empty,
        # and `List::create(List())` for ({},0).
        # This suggests `[]` for (X,0) and `[[]]` for ({},0).
        self._partitions.append([]) # The partition is an empty list of parts.
      else:
        # Cannot partition a non-empty set into 0 parts. Result is no partitions.
        pass # self._partitions remains []
      return

    # From here, k > 0
    if self.n == 0: # and k > 0
      # Partition of an empty set into k > 0 parts.
      # This means k empty sub-multisets.
      self._partitions.append([[] for _ in range(self.k)])
      return

    # From here, n > 0 and k > 0
    if self.k > self.n:
      # Cannot partition n items into k > n non-empty parts.
      return

    # State for recursive generation
    # self._s[i] stores the part index (0 to k-1) for multiset[i]
    self._s = [-1] * self.n 
    self._generate_recursive(0, 0) # current_element_index, num_parts_filled

  def _generate_recursive(self, element_idx, num_parts_active):
    """
    Recursive helper to generate partitions.

    Args:
      element_idx: Current index in self.multiset being assigned to a part.
      num_parts_active: How many parts have at least one element assigned so far.
    """
    if element_idx == self.n:
      # All elements assigned. Check if all k parts were used.
      if num_parts_active == self.k:
        current_partition_temp = [[] for _ in range(self.k)]
        part_elements_count = [0] * self.k
        for i in range(self.n):
          part_idx = self._s[i]
          current_partition_temp[part_idx].append(self.multiset[i])
          part_elements_count[part_idx] +=1
        
        # Ensure all parts are non-empty (double check, though num_parts_active should ensure this)
        if all(count > 0 for count in part_elements_count):
          # Canonical form: sort elements within each part, then sort the parts themselves.
          # No need to sort individual elements here as they are added from sorted self.multiset
          # and appended. Sorting parts is crucial.
          # The parts in current_partition_temp are already formed correctly.
          # Sorting them makes the overall partition canonical.
          sorted_list_of_parts = sorted([sorted(part) for part in current_partition_temp])
          # Add if not already found (should not happen with correct logic but as safeguard)
          if sorted_list_of_parts not in self._partitions:
            self._partitions.append(sorted_list_of_parts)
      return

    # Determine the starting part index for the current element
    # This handles multiset property: if multiset[i] == multiset[i-1],
    # then s[i] >= s[i-1] to ensure canonical assignment of identical items.
    start_part_idx = 0
    if element_idx > 0 and self.multiset[element_idx] == self.multiset[element_idx - 1]:
      start_part_idx = self._s[element_idx - 1]

    # Try assigning current element (self.multiset[element_idx]) to a part
    for part_idx in range(start_part_idx, self.k):
      self._s[element_idx] = part_idx
      
      # Check if this assignment opens up a new part
      # To do this, we need to know how many parts were filled *before* this assignment.
      # This is complex. A simpler way: count distinct values in self._s up to element_idx.
      
      # The logic from referenced C++ (likely related to Knuth or similar):
      # An element can be placed in an existing part or a new part (if not all k parts are active yet).
      # If it's placed in part `j`, and `j` was previously empty, `num_parts_active` increases.
      # Max part index allowed to open: `num_parts_active`.
      # If `num_parts_active < k`, we can assign to `part[num_parts_active]` (0-indexed).
      # Or assign to any `part[j]` where `j < num_parts_active`.

      # Let `max_part_to_assign_to = min(num_parts_active, k - 1)`
      # If `num_parts_active < k`, we can also assign to part `num_parts_active` (effectively opening it).
      # So, iterate `part_idx` from `start_part_idx` up to `min(num_parts_active, k-1)`.
      # If `part_idx == num_parts_active` (and `num_parts_active < k`), this means we are opening a new part.

      newly_opened_part = False
      if part_idx == num_parts_active: # Trying to put element into a new, previously unused part slot
          if num_parts_active < self.k: # Only if there are part slots available
              newly_opened_part = True
          else: # Already k parts active, cannot open another one
              continue # This assignment is invalid (trying to use more than k parts)
      
      # If part_idx < num_parts_active, it's an existing active part.
      # If part_idx > num_parts_active, it's trying to assign to a part that's not yet active
      # and is not the *next* available one. This skips parts, e.g. {1}{}{2} for k=3.
      # This should be disallowed to ensure parts are filled contiguously.
      if part_idx > num_parts_active :
          continue

      self._generate_recursive(element_idx + 1, num_parts_active + (1 if newly_opened_part else 0))


  def get_partitions(self):
    """
    Returns the list of all generated partitions.
    Each partition is a list of lists (sub-multisets), sorted.
    """
    return self._partitions

def multiset_partitions(multiset, k):
  """
  Wrapper function to get partitions of a multiset.

  Args:
    multiset: A list or tuple representing the multiset.
    k: The number of non-empty parts in each partition.

  Returns:
    A list of partitions. Each partition is a list of lists (sub-multisets).
    Returns `[]` if no such partitions exist.
    For k=0 and empty multiset, returns `[[]]` (one partition: an empty list of parts).
    For k=0 and non-empty multiset, returns `[]`.
  """
  # Special handling for k=0 to match C++ output:
  if k == 0:
    return [[]] if not multiset else [] # [[]] for ({},0), [] for ({x},0)

  # For k > 0, MultisetPartitions class handles n=0 case appropriately.
  generator = MultisetPartitions(multiset, k)
  return generator.get_partitions()


def generate_partitions(p, q_max):
  """
  Generates partitions for polynomial exponentiation with equivalence check.

  Args:
    p: The number of variables (indices 0 to p-1).
    q_max: The maximum sum of exponents (q). Iterates q from 1 to q_max.

  Returns:
    A list of lists. Outer list indexed by q-1 (for q from 1 to q_max).
    Each inner list contains unique partitions for that q.
    A partition is represented as a tuple of tuples (sub-multisets),
    where sub-multisets are sorted tuples of variable indices.
    The list of partitions for a given q is sorted.
  """
  all_partitions_list = []

  for q_val in range(1, q_max + 1):
    # combinations_q represents multisets of variable indices for sum q_val
    # e.g., for p=2, q_val=2: [(0,0), (0,1), (1,1)]
    # These are the multisets (M) we need to partition.
    combinations_q = combinations_with_repetition(p, q_val)
    
    q_val_partitions_storage = []
    seen_canonical_forms_for_q = set()

    for multiset_to_partition in combinations_q: # e.g., multiset_to_partition = (0,0)
      # Partition this multiset_to_partition into k_parts, where k_parts can range from 1 to q_val.
      # (A multiset of q_val items can be partitioned into at most q_val non-empty parts)
      for k_parts in range(1, q_val + 1):
        # partitions_of_current_multiset is list of (list of lists)
        # e.g., for multiset (0,0) and k_parts=1: [[[0,0]]]
        # e.g., for multiset (0,0) and k_parts=2: [[[0],[0]]]
        partitions_of_current_multiset = multiset_partitions(list(multiset_to_partition), k_parts)

        for single_partition in partitions_of_current_multiset:
          # single_partition is a list of lists (sub-multisets), e.g., [[0],[0]]
          # This partition is already canonical in terms of content of sub-multisets and their order.

          # Equivalence Check (from C++: sort counts of variables for each sub-multiset)
          count_vectors = []
          for sub_multiset in single_partition: # e.g., sub_multiset = [0]
            counts = [0] * p
            for var_idx in sub_multiset:
              if 0 <= var_idx < p: # Should always be true by construction from combinations_with_repetition
                counts[var_idx] += 1
            count_vectors.append(tuple(counts)) # Use tuple for hashability
          
          # Sort the count vectors to get a canonical form for the set of variable counts
          sorted_count_vectors = tuple(sorted(count_vectors))

          if sorted_count_vectors not in seen_canonical_forms_for_q:
            seen_canonical_forms_for_q.add(sorted_count_vectors)
            
            # Store the original partition (formatted as tuple of tuples)
            formatted_partition = tuple(tuple(sorted(sub)) for sub in single_partition)
            q_val_partitions_storage.append(formatted_partition)
            
    all_partitions_list.append(sorted(q_val_partitions_storage)) # Sort for deterministic output

  return all_partitions_list


def generate_partitions_full(p, q_max):
  """
  Generates partitions for polynomial exponentiation without equivalence check.
  (Equivalent to C++ generate_partitions_full).

  Args:
    p: The number of variables (indices 0 to p-1).
    q_max: The maximum sum of exponents (q). Iterates q from 1 to q_max.

  Returns:
    A list of lists, similar to generate_partitions. Partitions are not
    filtered by the count vector equivalence check.
  """
  all_partitions_list = []

  for q_val in range(1, q_max + 1):
    combinations_q = combinations_with_repetition(p, q_val)
    q_val_partitions_storage = []

    for multiset_to_partition in combinations_q:
      for k_parts in range(1, q_val + 1):
        partitions_of_current_multiset = multiset_partitions(list(multiset_to_partition), k_parts)
        
        for single_partition in partitions_of_current_multiset:
          # Store the partition (formatted as tuple of tuples)
          formatted_partition = tuple(tuple(sorted(sub)) for sub in single_partition)
          q_val_partitions_storage.append(formatted_partition)
            
    all_partitions_list.append(sorted(q_val_partitions_storage))

  return all_partitions_list
