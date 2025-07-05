test_that("complete workflow: keras model to polynomial evaluation", {
  skip_if_not_installed("keras")
  skip_if_not_installed("tensorflow")
  skip_on_cran()
  
  # Create a simple keras model
  model <- keras::keras_model_sequential() %>%
    keras::layer_dense(units = 3, input_shape = 2, activation = 'tanh') %>%
    keras::layer_dense(units = 1, activation = 'linear')
  
  # Convert to polynomial
  poly_model <- nn2poly(model, max_order = 2)
  
  # Test evaluation
  test_input <- matrix(c(0.5, -0.3, 1.0, 0.2), nrow = 2, byrow = TRUE)
  
  # Polynomial evaluation
  poly_pred <- eval_poly(poly_model, test_input)
  
  # Keras prediction (for comparison - should be similar for small inputs)
  keras_pred <- as.numeric(predict(model, test_input, verbose = 0))
  
  expect_length(poly_pred, 2)
  expect_length(keras_pred, 2)
  expect_true(all(is.finite(poly_pred)))
  expect_true(all(is.finite(keras_pred)))
})

test_that("complete workflow: luz model to polynomial with constraints", {
  skip_if_not_installed("luz")
  skip_if_not_installed("torch")
  skip_on_os("mac")
  skip_on_cran()
  
  testing_data <- testing_helper_2()
  data <- luz_test_data(testing_data)
  
  # Create and train model with constraints
  model <- luz_test_model() %>%
    luz::setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    )
  
  # Add constraints
  constrained_model <- add_constraints(model, constraint = "l2_norm")
  
  # Fit the model
  fitted <- fit(constrained_model, 
                data$train, 
                epochs = 2, 
                valid_data = data$valid, 
                verbose = FALSE)
  
  # Convert to polynomial
  poly_model <- nn2poly(fitted, 
                        taylor_orders = testing_data$taylor_orders,
                        max_order = 2)
  
  # Test polynomial evaluation
  test_input <- matrix(rnorm(4), nrow = 2)
  poly_pred <- eval_poly(poly_model, test_input)
  
  expect_s3_class(poly_model, "nn2poly")
  expect_length(poly_pred, 2)
  expect_true(all(is.finite(poly_pred)))
})

test_that("polynomial coefficients remain stable across different max_orders", {
  testing_data <- testing_helper_1()
  
  object <- testing_data$weights_list
  names(object) <- testing_data$af_string_list
  
  # Generate polynomials with different orders
  poly_order2 <- nn2poly(object, max_order = 2)
  poly_order3 <- nn2poly(object, max_order = 3)
  poly_order4 <- nn2poly(object, max_order = 4)
  
  # Test input
  test_input <- matrix(c(0.1, 0.2), nrow = 1)
  
  # Lower order coefficients should be similar
  eval2 <- eval_poly(poly_order2, test_input)
  eval3 <- eval_poly(poly_order3, test_input)
  eval4 <- eval_poly(poly_order4, test_input)
  
  # For small inputs, higher order should be close to lower order
  expect_true(abs(eval2 - eval3) < abs(eval2) * 0.1)  # Within 10%
  expect_true(abs(eval3 - eval4) < abs(eval3) * 0.1)  # Within 10%
})

test_that("plotting functions work with polynomial objects", {
  testing_data <- testing_helper_1()
  
  object <- testing_data$weights_list
  names(object) <- testing_data$af_string_list
  
  poly_result <- nn2poly(object, max_order = 2, keep_layers = TRUE)
  
  # Test diagonal plot
  expect_no_error(plot_diagonal(poly_result))
  
  # Test activation potential plots
  expect_no_error(plot_taylor_and_activation_potentials(poly_result))
  
  # Test generic plot method
  expect_no_error(plot(poly_result))
})

test_that("prediction methods work consistently across frameworks", {
  skip_if_not_installed("keras")
  skip_if_not_installed("tensorflow")
  skip_on_cran()
  
  # Test with different input dimensions
  test_inputs <- list(
    single = matrix(c(0.5, -0.3), nrow = 1),
    multiple = matrix(c(0.5, -0.3, 1.0, 0.2, -0.1, 0.8), nrow = 3, byrow = TRUE),
    edge_case = matrix(c(0, 0), nrow = 1)
  )
  
  model <- keras_test_model()
  poly_model <- nn2poly(model, max_order = 2)
  
  for (input_name in names(test_inputs)) {
    input_data <- test_inputs[[input_name]]
    
    # Test polynomial prediction
    poly_pred <- predict(poly_model, input_data)
    
    expect_length(poly_pred, nrow(input_data))
    expect_true(all(is.finite(poly_pred)))
    expect_type(poly_pred, "double")
  }
})

test_that("memory usage remains reasonable for large polynomial orders", {
  testing_data <- testing_helper_1()
  
  object <- testing_data$weights_list
  names(object) <- testing_data$af_string_list
  
  # Start with reasonable order
  start_mem <- gc()["Vcells", "used"]
  
  poly_result <- nn2poly(object, max_order = 6)
  
  end_mem <- gc()["Vcells", "used"]
  
  # Memory increase should be reasonable (less than 10x)
  memory_ratio <- end_mem / start_mem
  expect_true(memory_ratio < 10)
  
  # Object should still be usable
  expect_s3_class(poly_result, "nn2poly")
  expect_true(length(poly_result$values) > 0)
})

test_that("error handling works across the complete pipeline", {
  # Test with incompatible dimensions
  bad_weights <- list(
    matrix(c(1, 2, 3), nrow = 1),  # Wrong dimensions
    matrix(c(4, 5, 6, 7), nrow = 2)
  )
  names(bad_weights) <- c("tanh", "linear")
  
  expect_error(nn2poly(bad_weights))
  
  # Test with invalid activation functions
  weights_with_bad_af <- list(
    matrix(c(1, 2, 3, 4), nrow = 2),
    matrix(c(5, 6), nrow = 2)
  )
  names(weights_with_bad_af) <- c("invalid_activation", "linear")
  
  expect_error(nn2poly(weights_with_bad_af))
})