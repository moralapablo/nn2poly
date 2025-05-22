import pytest
import collections # For Counter in multiset_partitions tests
from nn2poly_py import combinatorics, helpers, algorithms

# --- Helper for combinations_with_repetition tests ---
def canonicalize_cwr_output(list_of_tuples):
    """Sorts each tuple then sorts the list of tuples."""
    return sorted([tuple(sorted(t)) for t in list_of_tuples])

# --- Helper for multiset_partitions tests ---
def canonicalize_msp_output(list_of_partitions):
    """
    A partition is a list of terms (sub-multisets/tuples).
    Sorts elements within each term, makes term a tuple.
    Sorts terms within each partition, makes partition a tuple.
    Sorts the list of partitions.
    e.g. from [[1,1],[2]] to ( ((1,1),(2,)) ) then list & sort.
    """
    canonical_partitions = []
    if not list_of_partitions:
        return []
    # Handle cases like [[]] for multiset_partitions([],0)
    if list_of_partitions == [[]]:
        return [()] # Represents one partition: the empty set of parts

    for partition in list_of_partitions:
        if not partition : # Handles partition being an empty list e.g. for k=0
             if not list_of_partitions[0] : # if list_of_partitions was [[]]
                 canonical_partitions.append(tuple()) # empty partition
                 continue

        # Ensure partition is not just a single empty list like [[]] for specific cases (e.g. msp([],1))
        # which should be (( (), ),)
        if len(partition) == 1 and not partition[0] and isinstance(partition[0], list):
             canonical_partitions.append( ( (), ) ) # Partition of {} into 1 part is {{}}
             continue
        if len(partition) > 1 and all(not sub_part and isinstance(sub_part, list) for sub_part in partition):
            # e.g. [[],[]] for msp([],2) -> ( ((),()), )
            canonical_partitions.append( tuple( [()] * len(partition) ) )
            continue


        sorted_terms_in_partition = tuple(sorted([tuple(sorted(term)) for term in partition]))
        canonical_partitions.append(sorted_terms_in_partition)
    return sorted(canonical_partitions)


# --- Helper for generate_partitions tests (comparing lists of ranked partitions) ---
def canonicalize_list_of_ranked_partitions(list_of_ranked_parts):
    """
    Each partition is typically a tuple of ranked terms (tuples of ranks).
    Sorts ranked terms within each partition, then sorts the list of partitions.
    """
    canonical_list = []
    for ranked_partition_tuple in list_of_ranked_parts:
        # Ensure terms are tuples and sorted, then sort the list of terms.
        # obtain_partitions_with_labels already produces tuple of sorted tuples of sorted ranks.
        # So, just need to sort the terms within the partition tuple.
        sorted_terms = tuple(sorted(list(ranked_partition_tuple)))
        canonical_list.append(sorted_terms)
    return sorted(canonical_list)


# === Tests for combinations_with_repetition ===

def test_combinations_with_repetition_5_3():
    expected_r_5_3_raw = [
        (1,1,1), (1,1,2), (1,1,3), (1,1,4), (1,1,5), (1,2,2), (1,2,3), 
        (1,2,4), (1,2,5), (1,3,3), (1,3,4), (1,3,5), (1,4,4), (1,4,5), 
        (1,5,5), (2,2,2), (2,2,3), (2,2,4), (2,2,5), (2,3,3), (2,3,4), 
        (2,3,5), (2,4,4), (2,4,5), (2,5,5), (3,3,3), (3,3,4), (3,3,5), 
        (3,4,4), (3,4,5), (3,5,5), (4,4,4), (4,4,5), (4,5,5), (5,5,5)
    ]
    expected_py_5_3 = canonicalize_cwr_output(expected_r_5_3_raw)
    
    actual_0_based = combinatorics.combinations_with_repetition(n=5, k=3)
    actual_1_based = [tuple(x + 1 for x in combo) for combo in actual_0_based]
    
    assert canonicalize_cwr_output(actual_1_based) == expected_py_5_3
    assert len(actual_1_based) == 35

def test_combinations_with_repetition_3_5():
    expected_r_3_5_raw = [
        (1,1,1,1,1), (1,1,1,1,2), (1,1,1,1,3), (1,1,1,2,2), (1,1,1,2,3),
        (1,1,1,3,3), (1,1,2,2,2), (1,1,2,2,3), (1,1,2,3,3), (1,1,3,3,3),
        (1,2,2,2,2), (1,2,2,2,3), (1,2,2,3,3), (1,2,3,3,3), (1,3,3,3,3),
        (2,2,2,2,2), (2,2,2,2,3), (2,2,2,3,3), (2,2,3,3,3), (2,3,3,3,3),
        (3,3,3,3,3)
    ]
    expected_py_3_5 = canonicalize_cwr_output(expected_r_3_5_raw)

    actual_0_based = combinatorics.combinations_with_repetition(n=3, k=5)
    actual_1_based = [tuple(x + 1 for x in combo) for combo in actual_0_based]

    assert canonicalize_cwr_output(actual_1_based) == expected_py_3_5
    assert len(actual_1_based) == 21

def test_combinations_with_repetition_edge_cases():
    assert combinatorics.combinations_with_repetition(n=5, k=0) == [()]
    assert combinatorics.combinations_with_repetition(n=0, k=5) == []
    assert combinatorics.combinations_with_repetition(n=0, k=0) == [()]


# === Tests for multiset_partitions ===

def test_multiset_partitions_simple_cases():
    # multiset = {1,1,2}, num_parts = 1 -> expected [ ((1,1,2),) ]
    res1 = combinatorics.multiset_partitions([1,1,2], 1)
    assert canonicalize_msp_output(res1) == sorted([ ((1,1,2),) ])

    # multiset = {1,1,2}, num_parts = 2 -> expected [ ((1,1),(2,)), ((1,),(1,2)) ] (order of partitions sorted)
    res2 = combinatorics.multiset_partitions([1,1,2], 2)
    expected2 = sorted([ ((1,1),(2,)), ((1,),(1,2)) ])
    assert canonicalize_msp_output(res2) == expected2
    
    # multiset = {1,1,1}, num_parts = 2 -> expected [ ((1,),(1,1)) ]
    res3 = combinatorics.multiset_partitions([1,1,1], 2)
    assert canonicalize_msp_output(res3) == sorted([ ((1,),(1,1)) ])

    # multiset = {1,1,1}, num_parts = 3 -> expected [ ((1,),(1,),(1,)) ]
    res4 = combinatorics.multiset_partitions([1,1,1], 3)
    assert canonicalize_msp_output(res4) == sorted([ ((1,),(1,),(1,)) ])
    
    res5 = combinatorics.multiset_partitions([1,1], 3) # Cannot partition {1,1} into 3 non-empty parts
    assert canonicalize_msp_output(res5) == []

def test_multiset_partitions_edge_cases():
    # Test k=0 cases (matches R behavior based on wrapper)
    assert combinatorics.multiset_partitions([], 0) == [[]] # R: List::create(List())
    assert combinatorics.multiset_partitions([1], 0) == []  # R: Rcpp::List()

    # Test n=0, k>0 cases
    # R: multiset_partitions({},1) -> List::create(List::create()) -> [[[]]]
    # Python: MultisetPartitions([],1).get_partitions() -> [[[]]]
    # canonicalize_msp_output needs to handle this specific structure.
    # Expected output: one partition, which contains one empty sub-multiset (term).
    res_n0_k1 = combinatorics.multiset_partitions([], 1)
    assert canonicalize_msp_output(res_n0_k1) == sorted([ ( ((),) ) ])

    # R: multiset_partitions({},2) -> List::create(List::create(), List::create()) -> [[[],[]]]
    # Python: MultisetPartitions([],2).get_partitions() -> [[[], []]]
    # Expected: one partition, containing two empty sub-multisets.
    res_n0_k2 = combinatorics.multiset_partitions([], 2)
    assert canonicalize_msp_output(res_n0_k2) == sorted([ ( ((),()), ) ])


# === Tests for helpers.obtain_partitions_with_labels ===
# This function's output structure is: {'labels': list_of_eq_keys, 'partitions': list_of_lists_of_partitions}
# Each eq_key is tuple of 0-based ranks, e.g. (0,), (0,0), (0,0,1)
# Each list_of_partitions corresponds to an eq_key. A partition is a tuple of terms (tuples of 0-based ranks).

def _parse_r_snapshot_simplified(snapshot_data_list_1_based_ranks):
    """
    Parses R snapshot data where labels and partitions are already in 1-based ranks.
    Converts them to 0-based ranks for keys and values.
    """
    parsed_map = {}
    for r_label_block in snapshot_data_list_1_based_ranks:
        # First partition, first term assumed to be the R label (1-based ranks)
        # This is an assumption about the R snapshot structure provided.
        # If the snapshot is more complex, this extraction needs to be more robust.
        # For "generate_partitions(P,Q)" snapshots, the "label" is implicit per block.
        # The R snapshot for generate_partitions(5,3) has items like "[[idx]]".
        # The label for "[[idx]]" is `partitions[[idx]][[1]][[1]]` in R code.
        # This label is already a canonical form using 1-based ranks.
        
        # For this simplified parser, assume r_label_block[0][0] IS the 1-based rank label.
        r_label_1_based_ranks = tuple(r_label_block[0][0]) 
        eq_key_0_based_ranks = tuple(r - 1 for r in r_label_1_based_ranks)
        
        expected_partitions_for_this_key_0_based_ranks = []
        for r_partition_1_based_ranks in r_label_block:
            # r_partition_1_based_ranks is like [[1,1], [2]] (1-based ranks)
            terms_0_based_ranks = []
            for r_term_1_based_ranks in r_partition_1_based_ranks:
                # Convert term to 0-based ranks and sort for canonical term form
                term_0_based = tuple(sorted([r_val - 1 for r_val in r_term_1_based_ranks]))
                terms_0_based_ranks.append(term_0_based)
            
            # Sort the list of terms to make the partition canonical
            expected_partitions_for_this_key_0_based_ranks.append(tuple(sorted(terms_0_based_ranks)))
        
        # Sort all partitions for this key
        parsed_map[eq_key_0_based_ranks] = sorted(expected_partitions_for_this_key_0_based_ranks)
        
    return parsed_map

# R snapshot data for generate_partitions(p=2, q_max=2), assuming labels and partitions are in 1-based ranks
# Label (1,) (rank 0): Partitions: [ ( (0,), ) ]
# Label (2,) (rank 0): This is for a different original variable, but maps to same eq_key (0,)
#   The helpers.obtain_partitions_with_labels generates unique eq_keys based on p_vars and q_max.
#   For p=2, q=1: original (0,)=var1 maps to eq_key (0,); original (1,)=var2 maps to eq_key (0,).
#   The R snapshot for "generate_partitions(P,Q)" (which is `generate_partitions_combinations` output)
#   should list partitions for unique *equivalent_label_keys* (0-based ranks).
#   So, for eq_key (0,): partitions are [ (( (0,), ),) ] (one term, which is rank 0)
#   For eq_key (0,0): partitions are [ (( (0,0), ),),  ( ((0,),(0,)), ) ]
#   For eq_key (0,1): partitions are [ (( (0,1), ),),  ( ((0,),(1,)), ) ]

# Snapshot: `generate_partitions(p=2, q_max=2)`
# Output from R's `generate_partitions_combinations(2,2,TRUE)`
# $labels: List of 3: (0), (0,0), (0,1) (represented as 1-based in R: (1), (1,1), (1,2))
# $partitions:
#   For (0,): [ tuple( ( (0,), ) ) ] -> R: [[[1]]]
#   For (0,0): [ tuple( ( (0,0), ) ), tuple( ( (0,),(0,) ) ) ] -> R: [[[1,1]], [[1],[1]]]
#   For (0,1): [ tuple( ( (0,1), ) ), tuple( ( (0,),(1,) ) ) ] -> R: [[[1,2]], [[1],[2]]]

R_SNAPSHOT_GP_P2_Q2_INTERPRETED = {
    (0,):   sorted([ (((0,),),) ]),  # R: label (1), partitions [[[1]]]
    (0,0): sorted([ (((0,0),),), (((0,),(0,)),) ]), # R: label (1,1), partitions [[[1,1]], [[1],[1]]]
    (0,1): sorted([ (((0,1),),), (((0,),(1,)),) ])  # R: label (1,2), partitions [[[1,2]], [[1],[2]]]
}

def test_obtain_partitions_with_labels_p2_q2():
    p, q_max = 2, 2
    expected_map = R_SNAPSHOT_GP_P2_Q2_INTERPRETED
    
    actual_data = helpers.obtain_partitions_with_labels(p, q_max, combinatorics, algorithms)
    actual_labels_list = actual_data['labels']
    actual_partitions_list = actual_data['partitions']
    
    # Build a map from actual output for easier lookup and canonicalization
    actual_map = {
        label: canonicalize_list_of_ranked_partitions(partitions)
        for label, partitions in zip(actual_labels_list, actual_partitions_list)
    }
    
    assert len(actual_map) == len(expected_map)
    for expected_eq_key, expected_ranked_partitions_sorted in expected_map.items():
        assert expected_eq_key in actual_map, f"Expected eq_key {expected_eq_key} not found in actual output."
        actual_ranked_partitions_for_key_sorted = actual_map[expected_eq_key]
        
        assert actual_ranked_partitions_for_key_sorted == expected_ranked_partitions_sorted, \
            f"Partitions mismatch for eq_key {expected_eq_key}.\n" \
            f"Actual (sorted): {actual_ranked_partitions_for_key_sorted}\n" \
            f"Expected (sorted): {expected_ranked_partitions_sorted}"


# Snapshot: `generate_partitions(p=5, q_max=3)`
# The R snapshot has 20 unique labels (equivalent_label_keys).
# Example: Label (1,1,2) in R snapshot means eq_key (0,0,1) using 0-based ranks.
# Its partitions are also in 1-based ranks. E.g., ((1,1),(2)) -> ((0,0),(1,)).
# For p=5, q_max=3, obtain_partitions_with_labels generates these canonical eq_keys:
# q=1: (0,) - 1 label
# q=2: (0,0), (0,1) - 2 labels
# q=3: (0,0,0), (0,0,1), (0,1,1), (0,1,2) - 4 labels
# Total should be 1+2+4 = 7 unique equivalent_label_keys for p>=3.
# The R snapshot `generate_partitions(5,3)` (output of `generate_partitions_combinations`) lists 20 items.
# This means the "labels" in the R snapshot are not the same as the `equivalent_label_key`s if p is large.
# The R snapshot labels are: (1), (2)..(5) for q=1. These map to one eq_key (0,).
# (1,1), (1,2)..(1,5), (2,2)..(5,5) for q=2. These map to eq_keys (0,0) and (0,1).
# This discrepancy means direct comparison with the R snapshot is very hard unless the snapshot
# is re-interpreted as "partitions for an *original monomial* that reduces to this canonical form".
# The prompt's description of `generate_partitions(5,3)` as "[[1]] label (1), [[6]] label (1,1)"
# implies the labels in the snapshot *are* these "original monomial forms" (or at least simple ones).
# And their partitions are listed.
# If `helpers.obtain_partitions_with_labels` produces `equivalent_label_key`s, then a mapping is needed.

# The current `helpers.obtain_partitions_with_labels` generates labels based on (p_vars, q_degree)
# then gets their `eq_key`, then generates partitions for the *original monomial*, then ranks those partitions.
# This matches the C++ `generate_partitions_combinations`.
# The `labels` field in its output *is* the list of unique `eq_key`s.
# The R snapshot's "labels" (like "(1)", "(1,1)", "(1,1,2)") ARE the `eq_key`s, but with 1-based numbers.

R_SNAPSHOT_GP_P5_Q3_LABEL_SUBSET_INTERPRETED = {
    # R Label (1) -> eq_key (0,)
    (0,): sorted([ (((0,),),) ]),
    # R Label (1,1) -> eq_key (0,0)
    (0,0): sorted([ (((0,0),),), (((0,),(0,)),) ]),
    # R Label (1,1,1) -> eq_key (0,0,0)
    (0,0,0): sorted([ (((0,0,0),),), (((0,0),(0,)),), (((0,),(0,),(0,)),) ]),
    # R Label (1,1,2) -> eq_key (0,0,1) (ranks for {varA, varA, varB})
    (0,0,1): sorted([
        (((0,0,1),),),                      # {AAB} -> {AAB}
        (((0,0),(1,)),),                    # {AA}{B}
        (((0,1),(0,)),),                    # {AB}{A}
        (((0,),(0,),(1,)),)                 # {A}{A}{B}
    ]),
    # R Label (1,2,3) -> eq_key (0,1,2) (ranks for {varA, varB, varC})
     (0,1,2): sorted([
        (((0,1,2),),),                      # {ABC}
        (((0,),(1,2)),),                    # {A}{BC}
        (((1,),(0,2)),),                    # {B}{AC}
        (((2,),(0,1)),),                    # {C}{AB}
        (((0,),(1,),(2,)),)                 # {A}{B}{C}
    ]),
}


def test_obtain_partitions_with_labels_p5_q3_selected_labels():
    # Test against selected labels from the R snapshot interpretation
    p, q_max = 5, 3 
    # Note: p=5 means original variables can be 0,1,2,3,4.
    # eq_key (0,0,1) means a pattern like x_i^2 * x_j
    # eq_key (0,1,2) means a pattern like x_i * x_j * x_k
    
    expected_map = R_SNAPSHOT_GP_P5_Q3_LABEL_SUBSET_INTERPRETED
    
    actual_data = helpers.obtain_partitions_with_labels(p, q_max, combinatorics, algorithms)
    actual_labels_list = actual_data['labels']
    actual_partitions_list = actual_data['partitions']

    actual_map = {
        label: canonicalize_list_of_ranked_partitions(partitions)
        for label, partitions in zip(actual_labels_list, actual_partitions_list)
    }
    
    for expected_eq_key, expected_ranked_partitions_sorted in expected_map.items():
        # Check if this eq_key (derived from R snapshot interpretation) exists in Python output
        if expected_eq_key not in actual_map:
            # This can happen if p is small, e.g. p=2, then (0,1,2) is not possible.
            # For p=5, all these keys (0,), (0,0), (0,0,0), (0,0,1), (0,1,2) should be generated.
            pytest.fail(f"Expected eq_key {expected_eq_key} (from R snapshot interpretation) "
                        f"not found in actual Python output labels: {list(actual_map.keys())}")

        actual_ranked_partitions_for_key_sorted = actual_map[expected_eq_key]
        
        assert actual_ranked_partitions_for_key_sorted == expected_ranked_partitions_sorted, \
            f"Partitions mismatch for eq_key {expected_eq_key}.\n" \
            f"Actual (sorted): {actual_ranked_partitions_for_key_sorted}\n" \
            f"Expected (sorted): {expected_ranked_partitions_sorted}"

    # Check total number of unique equivalent_label_keys generated for p=5, q_max=3
    # These are keys based on ranks, so independent of p if p is large enough for the ranks.
    # q=1: (0,) - 1
    # q=2: (0,0), (0,1) - 2
    # q=3: (0,0,0), (0,0,1), (0,1,1), (0,1,2) - 4
    # Total = 1 + 2 + 4 = 7 unique rank-based labels.
    # The R snapshot has 20 items because it lists partitions for specific *1-based variable combinations*
    # that reduce to these canonical rank forms.
    # obtain_partitions_with_labels is designed to match the C++ `generate_partitions_combinations`
    # which returns partitions for each unique *equivalent_label_key* (rank-based).
    # So, for p>=3, q_max=3, it should always generate these 7 unique rank-based labels.
    if p >= 3 :
         assert len(actual_labels_list) == 7, "Number of unique equivalent_label_keys should be 7 for p>=3, q_max=3."
    elif p == 2: # (0,1,2) is not possible
         assert len(actual_labels_list) == (1+2+3) # (0,0,0), (0,0,1), (0,1,1) = 6
    elif p == 1: # (0,1), (0,0,1), (0,1,1), (0,1,2) not possible
         assert len(actual_labels_list) == (1+1+1) # (0,), (0,0), (0,0,0) = 3
```
