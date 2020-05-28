#' Custom classier network
#' @param n_classes  enter number of classes.
#' @param num_rois enter number of roi's.
#' @param pooling_regions enter number of pooling regions.
#' @return creates a classifer network
#' @examples
#' @export

classifier_network <- function(n_classes, num_rois, pooling_regions) {

  keras::keras_model_custom(model_fn = function(self) {

    self$roi_pooling_conv <- layer_roi_pooling_conv(
      pool_size = pooling_regions,
      num_rois = num_rois
    )

    self$flatten <- keras::time_distributed(layer = keras::layer_flatten())

    self$fc1 <- keras::time_distributed(layer = layer_dense(
      units = 4096,
      activation = "relu"
    ))

    self$fc2 <- keras::time_distributed(layer = layer_dense(
      units = 4096,
      activation = "relu"
    ))

    self$dense_class <- keras::time_distributed(layer = layer_dense(
      units = n_classes,
      activation = "softmax",
      kernel_initializer = "zero"
    ))

    self$dense_regr <- keras::time_distributed(layer = layer_dense(
      units = 4 * (n_classes - 1),
      activation = "linear",
      kernel_initializer = "zero"
    ))


    function(x, mask = NULL) {
      out <- x %>%
        self$roi_pooling_conv() %>%
        self$flatten() %>%
        self$fc1() %>%
        self$fc2()

      list(
        dense_class = self$dense_class(out),
        dense_regrr = self$dense_regr(out)
      )
    }
  },
  name = "classifier"
  )
}
