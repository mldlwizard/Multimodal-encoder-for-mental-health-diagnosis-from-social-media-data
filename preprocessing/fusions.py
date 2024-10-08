
import torch
import torch.nn as nn
from torch.autograd import Variable


#------------------------- Concat ------------------------------------#

class Concat(nn.Module):
    """Concatenation of input data on dimension 1."""

    def __init__(self):
        """Initialize Concat Module."""
        super(Concat, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of Concat.
        
        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)


#--------------------------------- Tensor Fusion -------------------------------#

class TensorFusion(nn.Module):
    """
    Implementation of TensorFusion Networks.
    
    See https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py for more and the original code.
    """
    def __init__(self):
        """Instantiates TensorFusion Network Module."""
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if len(modalities) == 1:
            return modalities[0]

        mod0 = modalities[0]
        nonfeature_size = mod0.shape[:-1]

        m = torch.cat((Variable(torch.ones(
            *nonfeature_size, 1).type(mod0.dtype).to(mod0.device), requires_grad=False), mod0), dim=-1)
        for mod in modalities[1:]:
            mod = torch.cat((Variable(torch.ones(
                *nonfeature_size, 1).type(mod.dtype).to(mod.device), requires_grad=False), mod), dim=-1)
            fused = torch.einsum('...i,...j->...ij', m, mod)
            m = fused.reshape([*nonfeature_size, -1])

        return m


#----------------------------- Low Rank Tensor Fusion ------------------------------#

class LowRankTensorFusion(nn.Module):
    """
    Implementation of Low-Rank Tensor Fusion.
    
    See https://github.com/Justin1904/Low-rank-Multimodal-Fusion for more information.
    """

    def __init__(self, input_dims, output_dim, rank, flatten=True):
        """
        Initialize LowRankTensorFusion object.
        
        :param input_dims: list or tuple of integers indicating input dimensions of the modalities
        :param output_dim: output dimension
        :param rank: a hyperparameter of LRTF. See link above for details
        :param flatten: Boolean to dictate if output should be flattened or not. Default: True
        
        """
        super(LowRankTensorFusion, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        # low-rank factors
        self.factors = nn.ParameterList()
        for i, input_dim in enumerate(input_dims):
            factor = nn.Parameter(torch.Tensor(
                self.rank, input_dim+1, self.output_dim))
            nn.init.xavier_normal_(factor)
            self.factors.append(factor)
            # print("factor type: ", type(factor))
            # self.register_parameter(f'factor_{i}', factor)
            

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(
            torch.Tensor(1, self.output_dim))
        # init the fusion weights
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        # self.register_parameter('fusion_weights', self.fusion_weights)
        # self.register_parameter('fusion_bias', self.fusion_bias)


    def forward(self, modalities):
        """
        Forward Pass of Low-Rank TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        batch_size = modalities[0].shape[0]
        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        fused_tensor = 1
        for (modality, factor) in zip(modalities, self.factors):
            ones = Variable(torch.ones(batch_size, 1).type(
                modality.dtype), requires_grad=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            if self.flatten:
                modality_withones = torch.cat(
                    (ones, torch.flatten(modality, start_dim=1)), dim=1)
            else:
                modality_withones = torch.cat((ones, modality), dim=1)

            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(
            1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output


#-------------------------------------

class MultiplicativeInteractions2Modal(nn.Module):
    """Implements 2-way Modal Multiplicative Interactions."""
    
    def __init__(self, input_dims, output_dim, output, flatten=False, clip=None, grad_clip=None, flip=False):
        """
        :param input_dims: list or tuple of 2 integers indicating input dimensions of the 2 modalities
        :param output_dim: output dimension
        :param output: type of MI, options from 'matrix3D','matrix','vector','scalar'
        :param flatten: whether we need to flatten the input modalities
        :param clip: clip parameter values, None if no clip
        :param grad_clip: clip grad values, None if no clip
        :param flip: whether to swap the two input modalities in forward function or not
        
        """
        super(MultiplicativeInteractions2Modal, self).__init__()
        self.input_dims = input_dims
        self.clip = clip
        self.output_dim = output_dim
        self.output = output
        self.flatten = flatten
        if output == 'matrix3D':
            self.W = nn.Parameter(torch.Tensor(
                input_dims[0], input_dims[1], output_dim[0], output_dim[1]))
            nn.init.xavier_normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(
                input_dims[0], output_dim[0], output_dim[1]))
            nn.init.xavier_normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(
                input_dims[1], output_dim[0], output_dim[1]))
            nn.init.xavier_normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim[0], output_dim[1]))
            nn.init.xavier_normal_(self.b)

        # most general Hypernetworks as Multiplicative Interactions.
        elif output == 'matrix':
            self.W = nn.Parameter(torch.Tensor(
                input_dims[0], input_dims[1], output_dim))
            nn.init.xavier_normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0], output_dim))
            nn.init.xavier_normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(input_dims[1], output_dim))
            nn.init.xavier_normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim))
            nn.init.normal_(self.b)
        # Diagonal Forms and Gating Mechanisms.
        elif output == 'vector':
            self.W = nn.Parameter(torch.Tensor(input_dims[0], input_dims[1]))
            nn.init.xavier_normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(
                self.input_dims[0], self.input_dims[1]))
            nn.init.xavier_normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.b)
        # Scales and Biases.
        elif output == 'scalar':
            self.W = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.b)
        self.flip = flip
        if grad_clip is not None:
            for p in self.parameters():
                p.register_hook(lambda grad: torch.clamp(
                    grad, grad_clip[0], grad_clip[1]))

    def _repeatHorizontally(self, tensor, dim):
        return tensor.repeat(dim).view(dim, -1).transpose(0, 1)

    def forward(self, modalities):
        """
        Forward Pass of MultiplicativeInteractions2Modal.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if len(modalities) == 1:
            return modalities[0]
        elif len(modalities) > 2:
            assert False
        m1 = modalities[0]
        m2 = modalities[1]
        if self.flip:
            m1 = modalities[1]
            m2 = modalities[0]

        if self.flatten:
            m1 = torch.flatten(m1, start_dim=1)
            m2 = torch.flatten(m2, start_dim=1)
        if self.clip is not None:
            m1 = torch.clip(m1, self.clip[0], self.clip[1])
            m2 = torch.clip(m2, self.clip[0], self.clip[1])

        if self.output == 'matrix3D':
            Wprime = torch.einsum('bn, nmpq -> bmpq', m1,
                                  self.W) + self.V  # bmpq
            bprime = torch.einsum('bn, npq -> bpq', m1,
                                  self.U) + self.b    # bpq
            output = torch.einsum('bm, bmpq -> bpq', m2,
                                  Wprime) + bprime   # bpq

        # Hypernetworks as Multiplicative Interactions.
        elif self.output == 'matrix':
            Wprime = torch.einsum('bn, nmd -> bmd', m1,
                                  self.W) + self.V      # bmd
            bprime = torch.matmul(m1, self.U) + self.b      # bmd
            output = torch.einsum('bm, bmd -> bd', m2,
                                  Wprime) + bprime             # bmd

        # Diagonal Forms and Gating Mechanisms.
        elif self.output == 'vector':
            Wprime = torch.matmul(m1, self.W) + self.V      # bm
            bprime = torch.matmul(m1, self.U) + self.b      # b
            output = Wprime*m2 + bprime             # bm

        # Scales and Biases.
        elif self.output == 'scalar':
            Wprime = torch.matmul(m1, self.W.unsqueeze(1)).squeeze(1) + self.V
            bprime = torch.matmul(m1, self.U.unsqueeze(1)).squeeze(1) + self.b
            output = self._repeatHorizontally(
                Wprime, self.input_dims[1]) * m2 + self._repeatHorizontally(bprime, self.input_dims[1])
        return output
        