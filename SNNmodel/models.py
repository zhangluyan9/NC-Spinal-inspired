import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from units import *
import catCuda

# 4 *6 24*100 
#[0,1,0,1,0,1,0,...,1] T=4, 8, 16, 32, 64, 128*1us

#    T=4      8      16      32   64
# 1  acc+-[] 
# 1
# 1
# 2
# 3                              acc+-[] 
class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=2, quantize_level=7, dropout_rates=None):
        super(Net, self).__init__()
        
        # Handle dropout rates
        if dropout_rates is None:
            # Default: use 0.1 for all layers (backward compatibility)
            dropout_rates = [0.1] * len(hidden_sizes)
        elif isinstance(dropout_rates, (int, float)):
            # Single value: use same rate for all layers
            dropout_rates = [dropout_rates] * len(hidden_sizes)
        elif len(dropout_rates) != len(hidden_sizes):
            # Mismatch: extend or truncate to match hidden_sizes
            if len(dropout_rates) < len(hidden_sizes):
                # Extend with last value
                dropout_rates = dropout_rates + [dropout_rates[-1]] * (len(hidden_sizes) - len(dropout_rates))
            else:
                # Truncate to match
                dropout_rates = dropout_rates[:len(hidden_sizes)]
        
        self.hidden_layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes  
        for i in range(len(hidden_sizes)):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True),
                nn.BatchNorm1d(layer_sizes[i + 1]),
                Clamp_q_(q_level = quantize_level),
                nn.Dropout(p=dropout_rates[i])  # Use specific dropout rate for this layer
            ))
        
        self.fc_c = nn.Linear(hidden_sizes[-1], output_size, bias=False)

    def forward(self, x):
        x = torch.flatten(x, 1)
        #print("sssssss",x.shape)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.fc_c(x)
        return x
    
class InferenceNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=2,quantize_level =7):
        """
        Inference model without BatchNorm layers.
        :param input_size: Input feature dimension.
        :param hidden_sizes: List of hidden layer sizes.
        :param output_size: Output feature dimension.
        """
        super(InferenceNet, self).__init__()
        
        # Define the fully connected layers
        self.fc_layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)):
            self.fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        # Define the final output layer
        self.fc_out = nn.Linear(hidden_sizes[-1], output_size)

        # Activation
        self.cq = Clamp_q_(q_level = quantize_level)


    def forward(self, x):
        x = torch.flatten(x, 1)
        for fc in self.fc_layers:
            x = self.cq(fc(x))
            if "Linear(in_features=450, out_features=32, bias=True)"==str(fc):
                print(x)
        x = self.fc_out(x)
        return x

def round_to_nearest_power_of_two(threshold):
    # Ensure input is a tensor
    if not isinstance(threshold, torch.Tensor):
        threshold = torch.tensor(threshold, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate the rounded log2
    log2_rounded = torch.ceil(torch.log2(threshold))
    
    # Compute the nearest power of 2
    nearest_power_of_two = torch.pow(2, log2_rounded)
    
    return nearest_power_of_two
class Sparrow_SNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=2, quantized_index=32, T=31,Hybrid=False):
        """
        Spiking Neural Network (SNN) model.
        :param input_size: Input feature dimension.
        :param hidden_sizes: List of hidden layer sizes.
        :param output_size: Output feature dimension.
        :param quantized_index: Quantization index for weights and biases.
        :param T: Time window size.
        """
        super(Sparrow_SNN, self).__init__()

        self.T = T
        self.Hybrid = Hybrid
        self.quantized_index = quantized_index
        self.max_qw = 2 ** (quantized_index - 1) - 1
        self.min_qw = -2 ** (quantized_index - 1)
        self.act_max = 2 ** 8 -1
        self.cq = Clamp_q_(q_level = T)

        # Define the fully connected layers
        self.fc_layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)):
            self.fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True))

        # Define the final output layer
        self.fc_out = nn.Linear(hidden_sizes[-1], output_size, bias=True)

    def transform_and_sum(self, out, layer):
        """
        Apply transformation and summation for spiking computation.
        :param out: Input tensor.
        :param layer: Linear layer to apply.
        """

        out_ = out.clone().reshape(out.shape[0], out.shape[2], out.shape[1])
        for i in range(out.shape[0]):
            out_[i] = out[i].mT
        bs = out_.shape[0]
        out_reshaped = out_.view(-1, out_.shape[2])
        out_fc = layer(out_reshaped)
        out_s = out_fc.view(bs, self.T, -1)
        out_sum = torch.sum(out_s, dim=1)
        if layer != self.fc_out:
            spikes_data = [out_sum for _ in range(self.T)]
            out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda()
            threshold = self.T * 0.999 
            #threshold = floor_to_nearest_power_of_two(threshold)
            out = catCuda.getSpikes(out, torch.tensor(threshold))
            return out
        else:
            return out_sum
    
    def forward(self, x):

        for i, fc in enumerate(self.fc_layers):
            if i==0:
                x = torch.matmul(x, fc.weight.t())+fc.bias
                x = self.cq(x)
                spikes_data = [x for _ in range(self.T)]
                out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda()
                out = catCuda.getSpikes(out, 0.999)
                print(torch.sum(out))
            else:
                out = self.transform_and_sum(out, fc)
        #print(out)
        out_sum = self.transform_and_sum(out, self.fc_out)
        return out_sum

class SNN_LIF(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=2, quantized_index=8, T=31):
        """
        Spiking Neural Network (SNN) model.
        :param input_size: Input feature dimension.
        :param hidden_sizes: List of hidden layer sizes.
        :param output_size: Output feature dimension.
        :param quantized_index: Quantization index for weights and biases.
        :param T: Time window size.
        """
        super(SNN_LIF, self).__init__()

        self.T = T
        self.quantized_index = quantized_index

        # Define the fully connected layers
        self.fc_layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)):
            self.fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True))

        # Define the final output layer
        self.fc_out = nn.Linear(hidden_sizes[-1], output_size, bias=True)

    def transform_and_sum(self, out, layer):
        """
        Apply transformation and summation for spiking computation.
        :param out: Input tensor.
        :param layer: Linear layer to apply.
        """
        max_qw = 2 ** (self.quantized_index - 1) - 1
        min_wq = -2 ** (self.quantized_index - 1)
        out_ = out.clone().reshape(out.shape[0], out.shape[2], out.shape[1])
        for i in range(out.shape[0]):
            out_[i] = out[i].mT
        bs = out_.shape[0]
        out_reshaped = out_.view(-1, out_.shape[2])
        factor_max = max(torch.max(layer.weight), torch.max(layer.bias))
        factor_min = min(torch.min(layer.weight), torch.min(layer.bias))
        s = (factor_max - factor_min) / (max_qw - min_wq)
        layer.weight.data = torch.clamp(torch.round(layer.weight.data / s), min_wq, max_qw)
        layer.bias.data = torch.clamp(torch.round(layer.bias.data / s), min_wq, max_qw)
        out_fc = layer(out_reshaped)
        out_s = out_fc.view(bs, self.T,-1)
        x_transposed = out_s.transpose(1, 2).contiguous().cuda()
        if layer != self.fc_out:
            #spikes_data = [out_sum for _ in range(self.T)]
            #out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda()
            #print(out.shape)
            threshold =  0.999 / s
            out = catCuda.getSpikes(x_transposed, torch.floor(torch.tensor(threshold)))
            return out
        else:
            return torch.sum(x_transposed, dim=2)

    def forward(self, x):
        x_ = torch.flatten(x, 1)
        spikes_data = [x_ for _ in range(self.T)]
        out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda()
        out = catCuda.getSpikes(out, 0.999)
        #print(x_)
        #print(out)
        for layer in self.fc_layers:
            out = self.transform_and_sum(out, layer)
        #print(out)
        out_sum = self.transform_and_sum(out, self.fc_out)
        return out_sum
