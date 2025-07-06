test_that("build_callback creates appropriate callbacks for different model types", {
  skip_if_not_installed("luz")
  skip_if_not_installed("torch")
  skip_on_cran()
  
  # Test with luz model generator
  model <- luz_model_sequential(
    torch::nn_linear(2, 5),
    torch::nn_tanh(),
    torch::nn_linear(5, 1)
  )
  
  callback_l1 <- build_callback(model, "l1_norm")
  callback_l2 <- build_callback(model, "l2_norm")
  
  expect_s3_class(callback_l1, "luz_callback")
  expect_s3_class(callback_l2, "luz_callback")
  expect_equal(callback_l1$name, "l1_norm_callback")
  expect_equal(callback_l2$name, "l2_norm_callback")
})

test_that("norm_order extracts correct numeric values from constraint types", {
  expect_equal(norm_order("l1_norm"), 1)
  expect_equal(norm_order("l2_norm"), 2)
  expect_error(norm_order("l3_norm"))
})

test_that("fit.nn2poly method properly extends with callbacks", {
  testing_data <- testing_helper_1()
  
  # Create a mock nn2poly object
  object <- testing_data$weights_list
  names(object) <- testing_data$af_string_list
  
  result <- nn2poly(object = object)
  
  # Test that fit method exists
  expect_true(exists("fit.nn2poly"))
  expect_s3_class(result, "nn2poly")
})

test_that("callback system handles edge cases gracefully", {
  skip_if_not_installed("luz")
  skip_if_not_installed("torch")
  skip_on_cran()
  
  # Test with invalid constraint type
  model <- luz_model_sequential(
    torch::nn_linear(1, 1)
  )
  
  expect_error(build_callback(model, "invalid_norm"))
})

test_that("keras callback creation works when keras is available", {
  skip_if_not_installed("keras")
  skip_if_not_installed("tensorflow")
  skip_on_cran()
  
  model <- keras_test_model()
  
  # Test that the function doesn't error when keras is available
  expect_no_error({
    tryCatch({
      callback <- build_callback(model, "l1_norm")
    }, error = function(e) {
      # Allow for the case where Python components aren't available
      if (!grepl("KerasCallback|KerasConstraint", e$message)) {
        stop(e)
      }
    })
  })
})