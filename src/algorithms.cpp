// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
using namespace Rcpp;

// Helper function to find the index of a target label within a list of labels
// Returns -1 if not found (consider error handling for production code)
int find_label_index(const Rcpp::ListOf<Rcpp::IntegerVector>& all_labels, const Rcpp::IntegerVector& target_label) {
  for (int i = 0; i < all_labels.size(); ++i) {
    Rcpp::IntegerVector current_label = all_labels[i];
    if (current_label.size() == target_label.size()) {
      bool match = true;
      for (int j = 0; j < current_label.size(); ++j) {
        if (current_label[j] != target_label[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        return i;
      }
    }
  }
  return -1; // Not found
}

// [[Rcpp::export]]
arma::mat alg_non_linear(arma::mat coeffs_input,
                             ListOf<IntegerVector> labels_input,
                             ListOf<IntegerVector> labels_output,
                             IntegerVector taylor_orders, // Vector of Taylor orders for all layers
                             int current_layer, arma::vec g,
                             ListOf<IntegerVector> partitions_labels, // Pre-filtered labels
                             List partitions) // Pre-filtered partitions
{
  // Extract the needed parameters and values:
  // q_layer is the Taylor order for the current activation function
  int q_layer = taylor_orders[current_layer - 1];
  // q_previous_layer is no longer needed here for filtering, as partitions are pre-filtered.

  // Obtain total number of terms in the polynomial from labels
  int n_poly_terms = labels_output.size();

  // Obtain number of neurons
  int h_l = coeffs_input.n_rows;

  // We define the vector that will contain all the output coefficients
  arma::mat coeffs_output(h_l,n_poly_terms);

  ////////// Intercept //////////

  for (int n = 0; n <= q_layer; n++) {
    coeffs_output.col(0) = coeffs_output.col(0) + g[n] * arma::pow(coeffs_input.col(0), n);
    // we have to use g[n] to obtain g^(n)/n!,
    // because the function taylor already includes the term 1/n!
  }

  ////////// Rest of the coefficients //////////

  // As we already have all the coefficient labels, we can loop over them
  // Note that the intercept has to be skipped so start at 1
  for (int coeff_index = 1; coeff_index < n_poly_terms; coeff_index++) {
    IntegerVector label = labels_output[coeff_index];

    // Find the equivalence between label and a the ones needed for the
    // reduced partitions list
    std::multiset<int> mset(label.begin(), label.end());
    IntegerVector comp = unique(label).sort();
    IntegerVector mult(comp.size());
    for (int i = 0; i < comp.size(); i++)
      mult[i] = mset.count(comp[i]);
    comp = comp[order(mult, true)]; //decreasing
    IntegerVector seq = Range(1, comp.size());
    // IntegerVector equivalent_label =
    //   concat(seq, label)[match(label, concat(comp, label)) - 1];
    IntegerVector equivalent_label = match(label, comp);
    equivalent_label.sort();

    // Obtain chosen label position from the partitions labels list:
    int pos;
    for (pos = 0; pos < partitions_labels.size(); pos++) {
      if (partitions_labels[pos].size() != equivalent_label.size())
        continue;
      if (is_true(all(partitions_labels[pos] == equivalent_label)))
        break;
    }
    // If pos is out of bounds, it means the label was not found. This might indicate an issue.
    // For now, assume it's found. Consider adding error handling if pos == partitions_labels.size().

    // Extract the list with the (already filtered) partitions for the chosen label:
    // Construct as std::vector then wrap, to avoid Rcpp proxy object issues.
    List outer_list_sexp = as<List>(partitions[pos]); // partitions[pos] is a SEXP for ListOf<ListOf<IV>>
    std::vector<ListOf<IntegerVector>> temp_allowed_partitions_std_vec;
    temp_allowed_partitions_std_vec.reserve(outer_list_sexp.size());
    for (int k_outer = 0; k_outer < outer_list_sexp.size(); ++k_outer) {
        // Each element outer_list_sexp[k_outer] is a SEXP for ListOf<IV>
        temp_allowed_partitions_std_vec.push_back(as<ListOf<IntegerVector>>(outer_list_sexp[k_outer]));
    }
    ListOf<ListOf<IntegerVector>> allowed_partitions = wrap(temp_allowed_partitions_std_vec);

    // Number of partitions
    int n_allowed_partitions = allowed_partitions.size();

    // Replace again all the partitions to match the original indexes
    // This loop iterates through the pre-filtered partitions.
    for (int p_index = 0; p_index < n_allowed_partitions; p_index++) {
      // Explicitly use as<>() to convert from the SEXP obtained by operator[]
      // This helps resolve ambiguity with ChildVector proxies.
      ListOf<IntegerVector> aux = Rcpp::as<ListOf<IntegerVector>>(allowed_partitions[p_index]);
      for (int i = 0; i < aux.size(); i++) {
        IntegerVector auxv = aux[i];
        aux[i] = concat(comp, auxv)[match(auxv, concat(seq, auxv)) - 1];
        aux[i].sort();
      }
    }

    // Now, use the correctly renamed partitions
    for (int n = 1; n <= q_layer; n++) {

      arma::vec summatory(h_l);

      for (int p_index = 0; p_index < n_allowed_partitions; p_index++) {
        // Extract the chosen partition (a list) from the allowed partitions
        // Explicitly use as<>() to convert from the SEXP obtained by operator[]
        ListOf<IntegerVector> partition = Rcpp::as<ListOf<IntegerVector>>(allowed_partitions[p_index]);

        // We now need to check that each partition does not exceed n elements
        // so we have the condition m_0 + ... + m_C = n satisfied.
        // We also need the difference between the n_terms_in_partition
        // with respect to n, so we can add that difference as the exponent
        // of the intercept term. Then we compute this diff:
        int difference = n - partition.size();

        // If this diff is <0, we skip the partition
        // This is due to the second restriction to the allowed partitions, that
        // depends on n
        if (difference < 0) continue;

        // We need to obtain the m_index values to compute the multinomial
        // A. Replace Multinomial Coefficient Calculation
        std::map<std::vector<int>, int> unique_part_counts;
        for (int k = 0; k < partition.size(); ++k) {
          IntegerVector rcpp_vec = partition[k];
          std::vector<int> std_vec(rcpp_vec.begin(), rcpp_vec.end());
          unique_part_counts[std_vec]++;
        }

        std::vector<double> all_counts_for_lgamma;
        all_counts_for_lgamma.push_back(static_cast<double>(difference + 1.0)); // +1 for lgamma

        double sum_of_log_factorials = 0.0;
        sum_of_log_factorials += std::lgamma(static_cast<double>(difference + 1.0));

        for (const auto& entry : unique_part_counts) {
          sum_of_log_factorials += std::lgamma(static_cast<double>(entry.second + 1.0));
        }

        double multinomial_coef = std::exp(std::lgamma(n + 1.0) - sum_of_log_factorials);
        if (sum_of_log_factorials > std::lgamma(n + 1.0) + 1e-9) { // Check for potential negative result before exp due to precision
             multinomial_coef = 0; // if sum of part factorials is larger than n!
        }


        // B. Replace Product of Powered Coefficients Calculation
        arma::vec combined_term_product = arma::ones<arma::vec>(coeffs_input.n_rows);
        for (const auto& entry : unique_part_counts) {
          // Convert std::vector<int> key back to Rcpp::IntegerVector
          Rcpp::IntegerVector current_part_label_key = Rcpp::wrap(entry.first);

          int input_col_idx = find_label_index(labels_input, current_part_label_key);
          if (input_col_idx == -1) {
            // This should ideally not happen if labels_input is comprehensive
            // and partitions are derived correctly.
            Rcpp::stop("Label part not found in input labels during product calculation.");
          }
          arma::vec coeffs_column = coeffs_input.col(input_col_idx);
          combined_term_product %= arma::pow(coeffs_column, entry.second);
        }

        // C. Update Summatory
        summatory += multinomial_coef * combined_term_product % arma::pow(coeffs_input.col(0), difference);
        // Note that coeffs_input.col(0) is the intercept


      }
      // After the summatory over the partitions has been computed, we need to
      // get its result and multiply by the correspondent derivative value, and
      // add to the already stored values, here we are computing the summatory
      // over n.
      coeffs_output.col(coeff_index) =  coeffs_output.col(coeff_index) +  g[n] * summatory;
    }
  }
  return coeffs_output;
}
