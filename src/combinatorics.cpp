#include <Rcpp.h>
#include "multiset.h"
using namespace Rcpp;

// [[Rcpp::export]]
IntegerMatrix combinations_with_repetition(int n, int k) {
  int n_row = Rf_choose(n + k - 1, k);
  IntegerMatrix out(n_row, k);
  IntegerVector pos(k, 1);

  for (int row = 0; row < n_row; row++) {
    for (int i = k - 1; i >= 0; i--) {
      if (pos[i] > n) {
        pos[i - 1]++;
        for (int j = i; j < k; j++)
          pos[j] = pos[j - 1];
      }
    }
    out(row, _) = pos;
    pos[k - 1]++;
  }

  return out;
}

// [[Rcpp::export]]
List generate_partitions(int p, int q_max) {
  // Initialize the output
  List output;

  // Iterate over the degrees
  for (int i = 1; i <= q_max; i++) {
    // Generate all the combinations for the coeffs of the given order
    IntegerMatrix comb = combinations_with_repetition(p, i);

    // Iterate over those combinations obtaining its partitions when needed
    for (int j = 0; j < comb.nrow(); j++) {
      // Create a multiset from the combination
      IntegerVector mset_R = comb(j, _);
      std::multiset<int> mset(mset_R.begin(), mset_R.end());

      // Check if occurrences are sorted in descending order, which
      // eliminates the equivalent situations (i.e., 112 is equivalent to 223,
      // to 332, to 221 and so on)
      // This also ensures that each multiset is processed only once.
      bool sorted_descending = true;
      for (int k_check = 1; k_check < p; ++k_check) {
        if (mset.count(k_check) < mset.count(k_check + 1)) {
          sorted_descending = false;
          break;
        }
      }
      if (!sorted_descending) continue;

      // Generate partitions for the current multiset
      std::list<Partition<int>> partitions_for_mset_cpp;
      for (auto partition : multiset_partitions<int>(mset)) {
        partitions_for_mset_cpp.push_back(partition);
      }

      // Convert to Rcpp types
      std::vector<ListOf<IntegerVector>> temp_list_of_partitions_std_vec;
      temp_list_of_partitions_std_vec.reserve(partitions_for_mset_cpp.size());

      for (const auto& p_cpp : partitions_for_mset_cpp) {
        std::vector<IntegerVector> temp_partition_std_vec;
        temp_partition_std_vec.reserve(p_cpp.size());
        for (const auto& subset_cpp : p_cpp) {
          IntegerVector subset_R(subset_cpp.begin(), subset_cpp.end());
          std::sort(subset_R.begin(), subset_R.end()); // Sort for consistency
          temp_partition_std_vec.push_back(subset_R);
        }
        temp_list_of_partitions_std_vec.push_back(Rcpp::wrap(temp_partition_std_vec));
      }
      ListOf<ListOf<IntegerVector>> list_of_partitions_for_mset_R = Rcpp::wrap(temp_list_of_partitions_std_vec);

      // Add the multiset and its partitions to the output list
      output.push_back(Rcpp::List::create(
        Rcpp::Named("multiset") = clone(mset_R),
        Rcpp::Named("partitions") = list_of_partitions_for_mset_R
      ));
    }
  }

  return output;
}
