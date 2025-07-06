test_that("nn2poly handles very small weights without numerical issues", {
  # Create weights with very small values
  small_weights <- list(
    matrix(c(1e-10, 1e-10, 1e-10, 1e-10), nrow = 2),
    matrix(c(1e-10, 1e-10), nrow = 2)
  )
  names(small_weights) <- c("tanh", "linear")
  
  result <- nn2poly(small_weights, max_order = 2)
  
  expect_s3_class(result, "nn2poly")
  expect_true(all(is.finite(result$values)))
  expect_false(any(is.na(result$values)))
})

test_that("nn2poly handles very large weights appropriately", {
  # Create weights with large values
  large_weights <- list(
    matrix(c(0, 0, 100, 100), nrow = 2),
    matrix(c(0, 100), nrow = 2)
  )
  names(large_weights) <- c("tanh", "linear")
  
  result <- nn2poly(large_weights, max_order = 2)
  
  expect_s3_class(result, "nn2poly")
  expect_true(all(is.finite(result$values)))
})

test_that("nn2poly maintains precision with mixed scale weights", {
  # Create weights with mixed scales
  mixed_weights <- list(
    matrix(c(0.001, 1000, 0.5, -0.5), nrow = 2),
    matrix(c(0.1, -100), nrow = 2)
  )
  names(mixed_weights) <- c("relu", "linear")
  
  result <- nn2poly(mixed_weights, max_order = 3)
  
  expect_s3_class(result, "nn2poly")
  expect_true(all(is.finite(result$values)))
  
  # Check that we don't lose significant digits
  expect_true(any(abs(result$values) > 1e-6))
})

test_that("eval_poly handles extreme input values gracefully", {
  testing_data <- testing_helper_1()
  
  object <- testing_data$weights_list
  names(object) <- testing_data$af_string_list
  
  poly_result <- nn2poly(object, max_order = 2)
  
  # Test with extreme input values
  extreme_inputs <- matrix(c(-1000, 1000), nrow = 1)
  large_inputs <- matrix(c(-100, 100), nrow = 1)
  tiny_inputs <- matrix(c(-1e-10, 1e-10), nrow = 1)
  
  # These should not error and should return finite values
  expect_no_error(result1 <- eval_poly(poly_result, large_inputs))
  expect_no_error(result2 <- eval_poly(poly_result, tiny_inputs))
  
  expect_true(all(is.finite(result1)))
  expect_true(all(is.finite(result2)))
})

test_that("nn2poly with high polynomial orders maintains stability", {
  testing_data <- testing_helper_1()
  
  object <- testing_data$weights_list
  names(object) <- testing_data$af_string_list
  
  # Test with high polynomial order
  result_high <- nn2poly(object, max_order = 5)
  
  expect_s3_class(result_high, "nn2poly")
  expect_true(all(is.finite(result_high$values)))
  
  # Check that coefficients don't become unreasonably large
  expect_true(all(abs(result_high$values) < 1e10))
})

test_that("taylor expansion handles activation functions at boundaries", {
  # Test with activation functions that have specific boundary behaviors
  boundary_weights <- list(
    matrix(c(0, 0, 0, 0), nrow = 2),  # Zero weights
    matrix(c(0, 0), nrow = 2)
  )
  names(boundary_weights) <- c("sigmoid", "linear")
  
  result <- nn2poly(boundary_weights, max_order = 3)
  
  expect_s3_class(result, "nn2poly")
  expect_true(all(is.finite(result$values)))
})

test_that("polynomial evaluation is consistent across different input ranges", {
  testing_data <- testing_helper_1()
  
  object <- testing_data$weights_list
  names(object) <- testing_data$af_string_list
  
  poly_result <- nn2poly(object, max_order = 2)
  
  # Test consistency across different input ranges
  inputs1 <- matrix(c(0.1, 0.2), nrow = 1)
  inputs2 <- matrix(c(1.0, 2.0), nrow = 1)
  inputs3 <- matrix(c(-0.5, 0.5), nrow = 1)
  
  eval1 <- eval_poly(poly_result, inputs1)
  eval2 <- eval_poly(poly_result, inputs2)
  eval3 <- eval_poly(poly_result, inputs3)
  
  # All evaluations should be finite and reasonable
  expect_true(all(is.finite(c(eval1, eval2, eval3))))
  expect_true(all(abs(c(eval1, eval2, eval3)) < 1e6))
})

test_that("nn2poly handles networks with zero biases correctly", {
  # Create weights with zero biases
  zero_bias_weights <- list(
    matrix(c(0, 0, 1, -1), nrow = 2),  # Zero bias, non-zero weights
    matrix(c(0, 0.5), nrow = 2)
  )
  names(zero_bias_weights) <- c("tanh", "linear")
  
  result <- nn2poly(zero_bias_weights, max_order = 2)
  
  expect_s3_class(result, "nn2poly")
  expect_true(all(is.finite(result$values)))
  
  # With zero bias and certain activations, some coefficients might be exactly zero
  # This is expected and should be handled correctly
})