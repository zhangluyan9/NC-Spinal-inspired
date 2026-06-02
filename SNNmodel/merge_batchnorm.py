from __future__ import print_function
import torch
import torch.nn as nn



def merge_fc_and_bn(sequential_layer):
    """
    Merges a fully connected (fc) layer and a batch normalization (bn) layer into a single fc layer.
    :param sequential_layer: A Sequential object containing an fc layer, bn layer, and activation
    :return: New weights and biases for the merged fc layer
    """
    # Extract layers from the Sequential
    fc_layer = sequential_layer[0]  # Linear layer
    bn_layer = sequential_layer[1]  # BatchNorm layer

    # Extract FC layer parameters
    W_fc = fc_layer.weight
    b_fc = fc_layer.bias if fc_layer.bias is not None else torch.zeros(W_fc.size(0)).cuda()

    # Extract BN layer parameters
    gamma = bn_layer.weight
    beta = bn_layer.bias
    running_mean = bn_layer.running_mean
    running_var = bn_layer.running_var
    eps = bn_layer.eps

    # Compute new weights and bias
    scale = gamma / torch.sqrt(running_var + eps)
    shift = beta - running_mean * scale

    W_new = W_fc * scale.unsqueeze(1)  # Scale weights
    b_new = b_fc * scale + shift  # Adjust bias

    return W_new, b_new


def transfer_weights_to_inference_model(trained_model, inference_model):
    """
    Transfers the weights from the trained model to the inference model after merging BN layers.
    """
    for trained_layer, inference_fc in zip(trained_model.hidden_layers, inference_model.fc_layers):
        # Merge the FC and BN layers from the Sequential
        W_new, b_new = merge_fc_and_bn(trained_layer)
        
        # Set the weights and biases for the inference model
        inference_fc.weight.data = W_new
        inference_fc.bias.data = b_new

    # Transfer the final layer
    inference_model.fc_out.weight.data = trained_model.fc_c.weight.data
    inference_model.fc_out.bias.data = trained_model.fc_c.bias.data if trained_model.fc_c.bias is not None else torch.zeros_like(inference_model.fc_out.bias)

    return inference_model
