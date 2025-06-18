#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
List filter_partitions_by_max_part_order(List global_partitions_obj, int max_part_order_filter) {
  List original_labels = global_partitions_obj["labels"];
  List original_partitions_list = global_partitions_obj["partitions"]; // This is a List of (ListOf<ListOf<IV>>)

  List filtered_partitions_list(original_partitions_list.size()); // Output list

  for (int i = 0; i < original_partitions_list.size(); ++i) {
    // Explicitly use as<>() for elements of original_partitions_list
    ListOf<ListOf<IntegerVector>> partitions_for_label = Rcpp::as<ListOf<ListOf<IntegerVector>>>(original_partitions_list[i]);

    std::vector<ListOf<IntegerVector>> temp_current_filtered_std_vec;
    // Reserve capacity if partitions_for_label.size() is known and large
    // temp_current_filtered_std_vec.reserve(partitions_for_label.size());

    for (int j = 0; j < partitions_for_label.size(); ++j) {
      // Explicitly use as<>() for elements of partitions_for_label
      ListOf<IntegerVector> single_partition = Rcpp::as<ListOf<IntegerVector>>(partitions_for_label[j]);
      bool keep_partition = true;
      for (int k = 0; k < single_partition.size(); ++k) {
        // Explicitly use as<>() for elements of single_partition
        IntegerVector part_in_partition = Rcpp::as<IntegerVector>(single_partition[k]);
        if (part_in_partition.length() > max_part_order_filter) {
          keep_partition = false;
          break;
        }
      }
      if (keep_partition) {
        temp_current_filtered_std_vec.push_back(single_partition); // single_partition is already ListOf<IV>
      }
    }
    filtered_partitions_list[i] = Rcpp::wrap(temp_current_filtered_std_vec);
  }

  return List::create(
    _["labels"] = original_labels,
    _["partitions"] = filtered_partitions_list
  );
}
