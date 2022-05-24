

# Training settings:
* mode: network training mode
    - Regression
    - Classification
* backend: backend used in the current network
    - aefef 
    - mobilenet_v3_small
    - mobilenet_v3_large
    - pilotnet
    - resnet34
    - vgg16
    - smallervggnet
* fc_head: head used to compute the final output
    - custom_head
    - mobilenet_v3_large
    - mobilenet_v3_large_custom
    - pilotnet
    - smallervggnet
* loss_reduction: loss reduction mode used for MSELoss (only for regression)
    - mean
    - sum
* pretrained: flag to use pretrained weights.
* weight_loss: flag to weight the loss applied to each class (only for classification)
* apply_vertical_flip: flag to duplicate the entire training set applying a vertical flip
* non_common_samples_mult_factor: number to times a non-common sample will appear in the training dataset (non-common is assumed with |w| > 1)
* normalize_input: normalizes the input using the mean and std of the training values
* dataset: the dataset used to train the network
    - tfm
    - new
* base_size: image input size for the network
* batch_size: number of samples included in each batch during training. 