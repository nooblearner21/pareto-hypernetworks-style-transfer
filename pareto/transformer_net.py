import torch.nn.functional as F
from torch import nn
import torch

"""
Based on https://arxiv.org/pdf/2010.04104.pdf
"""
class Hyper(nn.Module):
    def __init__(
        self,
        preference_dim=2,
        preference_embedding_dim=32,
        hidden_dim=100,
        num_chunks=24,
        chunk_embedding_dim=64,
        num_hypervecs=7,
        hypervec_dim=10000,
    ):

        super().__init__()
        self.preference_embedding_dim = preference_embedding_dim
        self.num_chunks = num_chunks

        #Create embeddiing vector for parameter chunks
        self.chunk_embedding_matrix = nn.Embedding(
            num_embeddings=num_chunks, embedding_dim=chunk_embedding_dim
        )

        #Create embedding vector for preference ray
        self.preference_embedding_matrix = nn.Embedding(
            num_embeddings=preference_dim, embedding_dim=preference_embedding_dim
        )

        #Preference embedding and chunk embedding vectors through MLP
        self.fc = nn.Sequential(
            nn.Linear(preference_embedding_dim + chunk_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        list_hypervec = [self._init_w((hypervec_dim, hidden_dim)) for _ in range(num_hypervecs)]
        self.ws = nn.ParameterList(list_hypervec)

        #Initialize embedding vectors
        torch.nn.init.normal_(
            self.preference_embedding_matrix.weight, mean=0.0, std=0.1
        )
        torch.nn.init.normal_(self.chunk_embedding_matrix.weight, mean=0.0, std=0.1)
        for w in self.ws:
            torch.nn.init.normal_(w, mean=0.0, std=0.1)

        #Target network layer parameter shapes and sizes to generate
        self.target_layers = {
            "conv1.weight": torch.Size([32,3,3,3]),
            "conv1.bias": torch.Size([32]),
            "conv2.weight": torch.Size([64,32,3,3]),
            "conv2.bias": torch.Size([64]),
            "conv3.weight": torch.Size([128,64,3,3]),
            "conv3.bias": torch.Size([128]),

            "res1.0.weight": torch.Size([128,128,3,3]),
            "res1.0.bias": torch.Size([128]),
            "res1.1.weight": torch.Size([128,128,3,3]),
            "res1.1.bias": torch.Size([128]),

            "res2.0.weight": torch.Size([128,128,3,3]),
            "res2.0.bias": torch.Size([128]),
            "res2.1.weight": torch.Size([128,128,3,3]),
            "res2.1.bias": torch.Size([128]),

            "res3.0.weight": torch.Size([128,128,3,3]),
            "res3.0.bias": torch.Size([128]),
            "res3.1.weight": torch.Size([128,128,3,3]),
            "res3.1.bias": torch.Size([128]),

            "res4.0.weight": torch.Size([128,128,3,3]),
            "res4.0.bias": torch.Size([128]),
            "res4.1.weight": torch.Size([128,128,3,3]),
            "res4.1.bias": torch.Size([128]),

            "res5.0.weight": torch.Size([128,128,3,3]),
            "res5.0.bias": torch.Size([128]),
            "res5.1.weight": torch.Size([128,128,3,3]),
            "res5.1.bias": torch.Size([128]),

            "up1.weight": torch.Size([64,128,3,3]),
            "up1.bias": torch.Size([64]),
            "up2.weight": torch.Size([32,64,3,3]),
            "up2.bias": torch.Size([32]),
            "output.weight": torch.Size([3,32,9,9]),
            "output.bias": torch.Size([3]),
        }

    #Initialize random matrices
    def _init_w(self, shapes):
        return nn.Parameter(torch.randn(shapes), requires_grad=True)

    def forward(self, preference):
        #Style-Content preference embedding lookup
        pref_embedding = torch.zeros(
            (self.preference_embedding_dim,), device=preference.device
        )

        #For each preference coordinate rate lookup embedding vector
        #and calculate weighted sum to final preference vector
        for i, pref in enumerate(preference):
            pref_embedding += (
                self.preference_embedding_matrix(
                    torch.tensor([i], device=preference.device)
                ).squeeze(0)
                * pref
            )

        #Chunk embedding
        weights = []
        for chunk_id in range(self.num_chunks):
            chunk_embedding = self.chunk_embedding_matrix(
                torch.tensor([chunk_id], device=preference.device)
            ).squeeze(0)
            #Run the chunk and preference embeddings through the MLP to find hidden representation
            input_embedding = torch.cat((pref_embedding, chunk_embedding)).unsqueeze(0)
            rep = self.fc(input_embedding)

            #Concat hypervectors to a long vector and append to weight matrix
            weights.append(torch.cat([F.linear(rep, weight=w) for w in self.ws], dim=1))

        weight_vector = torch.cat(weights, dim=1).squeeze(0)

        weights_dict = dict()
        position = 0
        for name, shapes in self.target_layers.items():
            weights_dict[name] = weight_vector[position : position + shapes.numel()].reshape(shapes)
            position += shapes.numel()
        return weights_dict


class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()

        self.conv1 = ConvLayer(3,32,3,1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.relu1 = nn.ReLU()

        """
        Downsampling
        """
        self.conv2 = ConvLayer(32,64,3,2)
        self.relu2 = nn.ReLU()
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64,128,3,2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.relu3 = nn.ReLU()


        """
        Resblocks
        """
        self.res1 = ResLayer(128, kernel=3, stride=1)
        self.res2 = ResLayer(128, kernel=3, stride=1)
        self.res3 = ResLayer(128, kernel=3, stride=1)
        self.res4 = ResLayer(128, kernel=3, stride=1)
        self.res5 = ResLayer(128, kernel=3, stride=1)


        """
        Upsampling
        """
        self.up1 = UpsampleLayer(128, 64, kernel=3, stride=1)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.relu4 = nn.ReLU()
        self.up2 = UpsampleLayer(64,32, kernel=3, stride=1)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.relu5 = nn.ReLU()


        """
        Output
        """
        self.conv_out = ConvLayer(32,3,9,1)


    def forward(self, x, weights):
        x = self.conv1(x, weights['conv1.weight'], weights['conv1.bias'])
        x = self.in1(x)
        x = self.relu1(x)
        x = self.conv2(x, weights["conv2.weight"], weights["conv2.bias"])
        x = self.in2(x)
        x = self.relu2(x)
        x = self.conv3(x, weights["conv3.weight"], weights["conv3.bias"])
        x = self.in3(x)
        x = self.relu3(x)


        x = self.res1(x, weights['res1.0.weight'], weights['res1.0.bias'], weights['res1.1.weight'], weights['res1.1.bias'])
        x = self.res2(x, weights['res2.0.weight'], weights['res2.0.bias'], weights['res2.1.weight'], weights['res2.1.bias'])
        x = self.res3(x, weights['res3.0.weight'], weights['res3.0.bias'], weights['res3.1.weight'], weights['res3.1.bias'])
        x = self.res4(x, weights['res4.0.weight'], weights['res4.0.bias'], weights['res4.1.weight'], weights['res4.1.bias'])
        x = self.res5(x, weights['res5.0.weight'], weights['res5.0.bias'], weights['res5.1.weight'], weights['res5.1.bias'])


        x = self.up1(x, weights['up1.weight'], weights['up1.bias'])
        x = self.in4(x)
        x = self.relu4(x)
        x = self.up2(x, weights['up2.weight'], weights['up2.bias'])
        x = self.in5(x)
        x = self.relu5(x)

        out = self.conv_out(x, weights['output.weight'], weights['output.bias'])
        return out


"""
Downsampling ConvLayer
"""
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()

        #Uses reflection padding to avoid border artifcats and disturbance
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)

    def forward(self, x, weights, biases):
        x = self.reflection_pad(x)
        out = F.conv2d(
            x,
            weight=weights.reshape(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
            ),
            bias=biases,
            stride=self.stride,
        )
        return out


"""
Residual Blocks With Instance Normalization
"""
class ResLayer(nn.Module):
    def __init__(self, channels, kernel, stride):
        super(ResLayer, self).__init__()
        self.kernel_size = kernel
        self.stride = stride

        self.reflection_pad1 = nn.ReflectionPad2d(kernel // 2)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu1 = nn.ReLU()
        self.reflection_pad2 = nn.ReflectionPad2d(kernel // 2)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)



    def forward(self, x, weights1, bias1, weights2, bias2):
        residual = x
        x = self.reflection_pad1(x)
        x = F.conv2d(
            x,
            weight=weights1.reshape(
                128, 128, self.kernel_size, self.kernel_size
            ),
            bias=bias1,
            stride=self.stride,
        )
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.reflection_pad2(x)
        x = F.conv2d(
            x,
            weight=weights2.reshape(
                128, 128, self.kernel_size, self.kernel_size
            ),
            bias=bias2,
            stride=self.stride,
        )
        x = self.norm2(x)

        return x + residual

"""
Upsampling Layer
Based on https://distill.pub/2016/deconv-checkerboard/
to avoid checkerboard upsampling 'effect' by using nearest neighbour interpolation
and a convoloution layer with dimension-preserving padding
"""
class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super(UpsampleLayer, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.in_channels = in_channels
        self.kernel_size = kernel

        self.upsample = nn.Upsample(scale_factor=2)
        self.reflection_pad = nn.ReflectionPad2d(kernel // 2)

    def forward(self, x, weights, biases):
        x = self.upsample(x)
        x = self.reflection_pad(x)
        out = F.conv2d(
            x,
            weight=weights.reshape(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
            ),
            bias=biases,
            stride=self.stride,
        )
        return out