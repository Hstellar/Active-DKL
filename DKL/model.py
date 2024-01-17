import torch.nn as nn

import torch

import gpytorch

from gpytorch.means import ConstantMean, LinearMean

from gpytorch.kernels import RBFKernel, ScaleKernel

from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution

from gpytorch.distributions import MultivariateNormal

from gpytorch.models import ApproximateGP, GP

from gpytorch.mlls import VariationalELBO, AddedLossTerm

from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP

from gpytorch.mlls import DeepApproximateMLL

import torch.nn.functional as F

from gpytorch.models import ApproximateGP

from gpytorch.variational import CholeskyVariationalDistribution

from gpytorch.variational import VariationalStrategy

 

class MLPModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super().__init__()

        self.input_fc = nn.Linear(input_dim, 1000)

        self.hidden_fc = nn.Linear(1000, hidden_dim)

        self.output_fc = nn.Linear(hidden_dim, output_dim)

 

    def forward(self, x):

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        h_1 = F.relu(self.input_fc(x))

        h_2 = F.relu(self.hidden_fc(h_1))

        y_pred = self.output_fc(h_2)

        return y_pred

 

class DilatedCausalConv1d(nn.Module):

    def __init__(self, out_channels: int, kernel_shape: int, dilation_factor: int, in_channels: int):

        super().__init__()

 

        def weights_init(m):

            if isinstance(m, nn.Conv1d):

                nn.init.kaiming_normal_(m.weight.data)

                nn.init.zeros_(m.bias.data)

 

        self.dilation_factor = dilation_factor

        self.dilated_causal_conv = nn.Conv1d(in_channels=in_channels,

                                             out_channels=out_channels,

                                             kernel_size=kernel_shape,

                                             dilation=dilation_factor)

        self.dilated_causal_conv.apply(weights_init)

 

        self.skip_connection = nn.Conv1d(in_channels=in_channels,

                                         out_channels=out_channels,

                                         kernel_size=1)

        self.skip_connection.apply(weights_init)

        self.leaky_relu = nn.LeakyReLU(0.1)

 

    def forward(self, x):

        x1 = self.leaky_relu(self.dilated_causal_conv(x))

        x2 = x[:, :, self.dilation_factor:]

        x2 = self.skip_connection(x)

        print(x1.shape)

        print(x2.shape)

        return x1 + x2

 

 

class WaveNet(nn.Module):

    def __init__(self, out_channels: int, kernel_shape: int, in_channels: int):

        super().__init__()

 

        def weights_init(m):

            if isinstance(m, nn.Conv1d):

                nn.init.kaiming_normal_(m.weight.data)

                nn.init.zeros_(m.bias.data)

 

        self.dilation_factors = [2 ** i for i in range(0, out_channels)]

        self.in_channels = [in_channels] + [out_channels for _ in range(out_channels)]

        self.dilated_causal_convs = nn.ModuleList(

            [DilatedCausalConv1d(out_channels, kernel_shape, self.dilation_factors[i], self.in_channels[i]) for i in

             range(out_channels)])

        for dilated_causal_conv in self.dilated_causal_convs:

            dilated_causal_conv.apply(weights_init)

 

        self.output_layer = nn.Conv1d(in_channels=self.in_channels[-1],

                                      out_channels=1,

                                      kernel_size=1)

        self.output_layer.apply(weights_init)

        self.leaky_relu = nn.LeakyReLU(0.1)

 

    def forward(self, x):

        for dilated_causal_conv in self.dilated_causal_convs:

            x = dilated_causal_conv(x)

        x = self.leaky_relu(self.output_layer(x))

        return x

 

 

class ConvRNN(nn.Module):

    def __init__(self, input_dim, timesteps, output_dim, kernel_size1=7, kernel_size2=5, kernel_size3=3,

                 n_channels1=32, n_channels2=32, n_channels3=32, n_units1=32, n_units2=32, n_units3=32):

        super().__init__()

        self.avg_pool1 = nn.AvgPool1d(2, 2)

        self.avg_pool2 = nn.AvgPool1d(4, 4)

        self.conv11 = nn.Conv1d(input_dim, n_channels1, kernel_size=kernel_size1)

        self.conv12 = nn.Conv1d(n_channels1, n_channels1, kernel_size=kernel_size1)

        self.conv21 = nn.Conv1d(input_dim, n_channels2, kernel_size=kernel_size2)

        self.conv22 = nn.Conv1d(n_channels2, n_channels2, kernel_size=kernel_size2)

        self.conv31 = nn.Conv1d(input_dim, n_channels3, kernel_size=kernel_size3)

        self.conv32 = nn.Conv1d(n_channels3, n_channels3, kernel_size=kernel_size3)

        self.gru1 = nn.GRU(n_channels1, n_units1, batch_first=True)

        self.gru2 = nn.GRU(n_channels2, n_units2, batch_first=True)

        self.gru3 = nn.GRU(n_channels3, n_units3, batch_first=True)

        self.linear1 = nn.Linear(n_units1 + n_units2 + n_units3, output_dim)

        self.linear2 = nn.Linear(input_dim * timesteps, output_dim)

        self.zp11 = nn.ConstantPad1d(((kernel_size1 - 1), 0), 0)

        self.zp12 = nn.ConstantPad1d(((kernel_size1 - 1), 0), 0)

        self.zp21 = nn.ConstantPad1d(((kernel_size2 - 1), 0), 0)

        self.zp22 = nn.ConstantPad1d(((kernel_size2 - 1), 0), 0)

        self.zp31 = nn.ConstantPad1d(((kernel_size3 - 1), 0), 0)

        self.zp32 = nn.ConstantPad1d(((kernel_size3 - 1), 0), 0)

 

    def forward(self, x):

        x = x.permute(0, 2, 1)

        # line1

        y1 = self.zp11(x)

        y1 = torch.relu(self.conv11(y1))

        y1 = self.zp12(y1)

        y1 = torch.relu(self.conv12(y1))

        y1 = y1.permute(0, 2, 1)

        out, h1 = self.gru1(y1)

        # line2

        y2 = self.avg_pool1(x)

        y2 = self.zp21(y2)

        y2 = torch.relu(self.conv21(y2))

        y2 = self.zp22(y2)

        y2 = torch.relu(self.conv22(y2))

        y2 = y2.permute(0, 2, 1)

        out, h2 = self.gru2(y2)

        # line3

        y3 = self.avg_pool2(x)

        y3 = self.zp31(y3)

        y3 = torch.relu(self.conv31(y3))

        y3 = self.zp32(y3)

        y3 = torch.relu(self.conv32(y3))

        y3 = y3.permute(0, 2, 1)

        out, h3 = self.gru3(y3)

        h = torch.cat([h1[-1], h2[-1], h3[-1]], dim=1)

        out1 = self.linear1(h)

        out2 = self.linear2(x.contiguous().view(x.shape[0], -1))

        out = out1 + out2

        return out

 

class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, device):

        super(LSTMModel, self).__init__()

        self.device = device

        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim

 

        # LSTM layers

        self.lstm = nn.LSTM(

            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, bias = True,# bidirectional =True

        )

        # self.transformer = transformer_model  = nn.Transformer(

        #     d_model=hidden_dim,#no. of features

        #     nhead=2,

        #     num_encoder_layers=2,

        #     num_decoder_layers=2,

        #     dim_feedforward=16,

        #     dropout=0.1,

        #     activation= "gelu",

        #     batch_first=True,

        #     norm_first=True,

        # )

        self.lstm2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        # Fully connected layer

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        # self.mean_module = gpytorch.means.ConstantMean()  # prior mean
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) # covariance

 

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device).requires_grad_()

        # h1 = torch.zeros(self.layer_dim, self.layer_dim, self.hidden_dim).to(device).requires_grad_()

        # c1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0, c0))
        #out, (hn1, cn1) = self.lstm2(out, (hn, cn))
        out = out[:, -1, :].to(self.device)
        # print('out shape',out.size())
        # out = self.transformer.encoder(out)
        # print('before',out.shape)
        out = self.fc(out)
        # print('after',out.shape)
        out = self.fc1(out)
        # out2 = self.fc1(x.contiguous().view(x.shape[0], -1))
        # out = out1 + out2
        
        return out

 

class  ExactGPLayer(gpytorch.models.ExactGP):

    def __init__(self,  train_x, train_y, likelihood, kernel):

        super( ExactGPLayer, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        ## RBF kernel

        if (kernel == 'rbf' or kernel == 'RBF'):

            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        ## Spectral kernel

        elif (kernel == 'spectral'):

            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,ard_num_dims=train_x.size(1))

        elif(kernel == 'cosine'):

            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())

        elif(kernel == 'BNCosSim'):

            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())+gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,ard_num_dims=train_x.size(1))

        elif(kernel == 'matern'):

            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=train_x.size(1)))

        else:

            raise ValueError(

                "[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

 

    def forward(self, x):

        mean_x = self.mean_module(x)

        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

 

 

 

class Bias(nn.Module):

    def __init__(self, input_dim, output_dim, device):

        super().__init__()

        self.device = device

        self.output_fc = nn.Linear(input_dim, output_dim)

 

    def forward(self, x):

        # batch_size = x.shape[0]

        # x = x.view(batch_size, -1)

        x = x.view(-1,1).to(self.device)

        y_pred = self.output_fc(x)

        return y_pred

 

 

class GPModel(ApproximateGP):

    def __init__(self, input_dims, output_dims, num_inducing=250):

        if output_dims is None:

            inducing_points = torch.randn(num_inducing, input_dims)

            batch_shape = torch.Size([])

        else:

            inducing_points = torch.randn(output_dims, num_inducing, input_dims)

            batch_shape = torch.Size([output_dims])

 

        variational_distribution = CholeskyVariationalDistribution(

            num_inducing_points=num_inducing,

            batch_shape=batch_shape

        )

 

        variational_strategy = VariationalStrategy(

            self,

            inducing_points,

            variational_distribution,

            learn_inducing_locations=True

        )

 

        super(GPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

 

    def forward(self, x):

        mean_x = self.mean_module(x)

        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

 

 

class ToyDeepGPHiddenLayer(DeepGPLayer):

    def __init__(self, input_dims, output_dims, num_inducing=250, mean_type='constant'):

        if output_dims is None:

            inducing_points = torch.randn(num_inducing, input_dims)

            batch_shape = torch.Size([])

        else:

            inducing_points = torch.randn(output_dims, num_inducing, input_dims)

            batch_shape = torch.Size([output_dims])

 

        variational_distribution = CholeskyVariationalDistribution(

            num_inducing_points=num_inducing,

            batch_shape=batch_shape

        )

 

        variational_strategy = VariationalStrategy(

            self,

            inducing_points,

            variational_distribution,

            learn_inducing_locations=True

        )

 

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

 

        if mean_type == 'constant':

            self.mean_module = ConstantMean(batch_shape=batch_shape)

        else:

            self.mean_module = LinearMean(input_dims)

        # k3 = gpytorch.kernels.SpectralDeltaKernel(num_dims=input_dims, num_deltas=50)

        # k3.initialize_from_data(train_x, train_y)

        self.covar_module = ScaleKernel(

            gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),

            # gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),

           # k3,

            batch_shape=batch_shape, ard_num_dims=None

        )

 

    def forward(self, x):

        mean_x = self.mean_module(x)

        covar_x = self.covar_module(x)

        return MultivariateNormal(mean_x, covar_x)

 

    def __call__(self, x, *other_inputs, **kwargs):

        """

        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections

        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first

        hidden layer's outputs and the input data to hidden_layer2.

        """

        if len(other_inputs):

            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):

                x = x.rsample()

 

            processed_inputs = [

                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)

                for inp in other_inputs

            ]

 

            x = torch.cat([x] + processed_inputs, dim=-1)

 

        return super().__call__(x, are_samples=bool(len(other_inputs)))

 

 

class DeepGP(DeepGP):

    def __init__(self, train_x_shape):

        hidden_layer0 = ToyDeepGPHiddenLayer(

            input_dims=train_x_shape[-1],

            output_dims=1,

            num_inducing =train_x_shape[0],

            mean_type='linear',

        )

 

        last_layer = ToyDeepGPHiddenLayer(

            input_dims=hidden_layer0.output_dims,

            output_dims=None,

            mean_type='constant',

        )

 

        super().__init__()

 

        self.hidden_layer0 = hidden_layer0

        # self.hidden_layer1 = hidden_layer1

        # self.hidden_layer2 = hidden_layer2

        self.last_layer = last_layer

        self.likelihood = GaussianLikelihood()

 

    def forward(self, inputs):

        hidden_rep0 = self.hidden_layer0(inputs)

        # hidden_rep1 = self.hidden_layer1(hidden_rep0)

        # hidden_rep2 = self.hidden_layer2(hidden_rep1)

        output = self.last_layer(hidden_rep0)

        return output

 

class ProbMCdropoutDNN(nn.Module):
    """
    Monte Carlo (MC) dropout neural network with 2 hidden layers.
    """

    def __init__(self, input_size, hidden_size_1=50, hidden_size_2=20, dropout=0.005):
        super(ProbMCdropoutDNN, self).__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size_1)
        self.linear2 = nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2)
        self.linear3 = nn.Linear(in_features=hidden_size_2, out_features=2)
        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # first linear layer
        x = self.linear1(x)
        x = self.softplus(x)
        x = self.dropout(x)

        # second linear layer
        x = self.linear2(x)
        x = self.softplus(x)
        x = self.dropout(x)

        x = self.linear3(x)
        return torch.distributions.Normal(
            loc=x[:, 0:1].squeeze(),
            scale=self.softplus(x[:, 1:2].squeeze()).add(other=1e-6)
        )

    def predict(self, x):
        distrs = self.forward(x)
        y_pred = distrs.sample()
        return y_pred

