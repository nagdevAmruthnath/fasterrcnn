#' Custom faster RCNN network
#' @param n_classes  enter number of classes. Ex 21
#' @param num_rois enter number of roi's. Ex 32
#' @param pooling_regions enter number of pooling regions. Ex 7
#' @param num_anchors enter number of anchors. Ex 4
#' @return creates a classifer network
#' @examples
#' # fasterrcnn(n_classes=4, num_rois=32, pooling_regions=7, num_anchors=4)
#' @export

fasterrcnn <- function(n_classes, num_rois,
                       pooling_regions, num_anchors) {
  keras::keras_model_custom(function(self) {

    self$vgg <- application_vgg16(
      include_top = FALSE,
      input_shape = shape(NULL, NULL, 3),
      weights = NULL
    )

    self$rpn <- region_proposal_network(num_anchors = num_anchors)

    self$cls <- classifier_network(
      n_classes = n_classes,
      num_rois = num_rois,
      pooling_regions = pooling_regions
    )

    function(x, masks = NULL) {
      img <- x[[1]]
      roi <- x[[2]]

      shared <- self$vgg(img)

      out_rpn <- self$rpn(shared)
      out_cls <- self$cls(list(shared, roi))

      append(out_rpn, out_cls)
    }
  })
}



