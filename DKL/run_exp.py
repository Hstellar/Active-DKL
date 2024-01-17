import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import pickle
import time
import logging
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans


class Optimization(nn.Module):
    def __init__(
        self,
        feature_extractor,
        model,
        likelihood,
        loss_fn,
        optimizer,
        # query_strategy,
        # n_cluster,
        device,
        mode,
    ):
        super(Optimization, self).__init__()
        self.feature_extractor = feature_extractor
        self.likelihood = likelihood
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.mode = mode
        # self.query_strategy = query_strategy
        # self.n_cluster = n_cluster
        self.train_losses = []
        self.val_losses = []
        self.val_mse = []
        self.scaler = torch.cuda.amp.GradScaler()

        # self.mll = #gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(self.device)

        self.mse = nn.MSELoss()
        self.mse_mean = []
        self.train_pred = []
    
    def coverage(y, yL, yH):
        return (100 / y.shape[0] * ((y>yL)&(y<yH)).sum())
    
    def eval_crps(self, pred, truth):
        """
        Evaluate continuous ranked probability score, averaged over all data
        elements.
        **References**
        [1] Tilmann Gneiting, Adrian E. Raftery (2007)
            `Strictly Proper Scoring Rules, Prediction, and Estimation`
            https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
        :param torch.Tensor pred: Forecasted samples.
        :param torch.Tensor truth: Ground truth.
        :rtype: float
        """

        opts = dict(device=pred.device, dtype=pred.dtype)
        num_samples = pred.size(0)
        pred = pred.sort(dim=0).values
        diff = pred[1:] - pred[:-1]

        weight = torch.arange(1, num_samples, **opts) * torch.arange(
            num_samples - 1, 0, -1, **opts
        )
        weight = weight.reshape(weight.shape + (1,) * truth.dim())
        return (((pred - truth).abs().mean(0) - (diff * weight).sum(0) / num_samples**2)
            .mean()
            .cpu()
            .item()
        )

    def train_step(self, x, y):
        # Sets model to train mode

        self.model.train()
        self.likelihood.train()
        self.feature_extractor.train()
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            with gpytorch.settings.cholesky_jitter(1e-1):
                if self.mode == "ExactGP":
                    # z = self.feature_extractor(x)
                    # self.model.set_train_data(inputs=x, targets=torch.flatten(y))
                    yhat = self.model(x)
                    loss = self.loss_fn(yhat, torch.flatten(y))

                elif self.mode == "DeepGP":
                    yhat = self.model(x)
                    loss = self.loss_fn(yhat, y.reshape(-1))

                elif self.mode == "LSTM_ExactGP":
                    z = self.feature_extractor(x).float().to(self.device)
                    
                    self.model.set_train_data(inputs=z, targets=torch.flatten(y))
                    yhat = self.model(z)

                    loss = -self.loss_fn(yhat, torch.flatten(y))

                else:
                    z = self.feature_extractor(x).to(self.device).float()
                    # print(z)
                    
                    self.model.set_train_data(inputs=z, targets=torch.flatten(y))
                    yhat = self.model(z)
                    loss = self.loss_fn(yhat, torch.flatten(y))

        loss.backward()
        # self.scaler.scale(loss).backward()
        # self.scaler.step(optimizer)
        self.optimizer.step()
        mse = self.mse(yhat.mean, y.view(1, -1))

        # Makes predictions
        # self.scaler.update()
        # Returns the loss
        return mse.item(), loss.item()


    def eval_step(self, batch_size, n_features, x_test, y_test):
        if (self.mode == "ExactGP") or (self.mode == "DeepGP"):
            x_test = x_test.view([batch_size, n_features]).to(self.device)

        else:
            x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
        # print('x_test',x_test.shape)
        # x_test = x_test.to(self.device)

        y_test = y_test.to(self.device)
        if self.mode == "ExactGP":
            z_query = x_test

        elif self.mode == "DeepGP":
            z_query = x_test

        elif self.mode == "LSTM_ExactGP":
            self.feature_extractor.eval()
            z_query = self.feature_extractor(x_test)

        else:
            self.feature_extractor.eval()
            z_query = self.feature_extractor(x_test)

        return z_query
    
    def train(self, train_loader,val_loader, batch_size=64, n_epochs=50, n_features=1):
        #         model_path = f'models/{self.model}'

        start_time = time.time()

        for epoch in range(1, n_epochs + 1):
            # q_points = self.get_query_points(batch_size, i)
            batch_losses = []
            batch_mse = []
            self.model.train()
            self.likelihood.train()
            self.feature_extractor.train()

            for x_batch, y_batch in train_loader:
                if (self.mode == "ExactGP") or (self.mode == "DeepGP"):
                    x_batch = x_batch.view([batch_size, n_features]).to(self.device)
                else:
                    x_batch = x_batch.view([batch_size, -1, n_features]).to(self.device)
                    y_batch = y_batch.to(self.device)
                
                # self.model.train()
                # self.likelihood.train()
                # self.feature_extractor.train()
                mse, loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
                batch_mse.append(mse)

            training_loss = np.mean(batch_losses)
            train_mse = np.mean(batch_mse)
            self.train_losses.append(training_loss)
            self.mse_mean.append(train_mse)

            if (epoch <= 10) | (epoch % 10 == 0):
                # \t val_mse: {validation_mse: .4f}\t validation cprs{val_cprs: .4f}")
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss: .4f} - noise = {self.likelihood.noise.item(): e} \t train_mse: {train_mse: .4f}"
                )

        with open("feature_extractor.pkl", "wb") as f:
            pickle.dump(self.feature_extractor, f)

        with open("train_losses.txt", "w") as output:
            output.write(str(self.train_losses))

        print("Wall clock(in hours):", (time.time() - start_time) / 3600)


    def evaluate(self, train_loader, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            self.model.eval()
            self.likelihood.eval()
            self.feature_extractor.eval()
            train_pred = []
            train_actual = []
            val_pred = []
            val_actual = []
            predictions = []
            values = []
            lower_pred = []
            upper_pred = []
            lower_f = []
            upper_f = []
            mean_f = []
            variance = []

            for x_test, y_test in test_loader:
                z_query = self.eval_step(batch_size, n_features, x_test, y_test)

                yhat = self.likelihood(self.model(z_query))

                # 2 standard deviations above and below the mean
                lower_p, upper_p = yhat.confidence_region()
                predictions.append(yhat.mean.detach().cpu().numpy())
                variance.append(yhat.variance.detach().cpu().numpy())
                values.append(y_test.detach().cpu().numpy())
                lower_pred.append(lower_p.detach().cpu().numpy())
                upper_pred.append(upper_p.detach().cpu().numpy())

                observed_f = self.model(z_query)
                mean_f.append(observed_f.mean.detach().cpu().numpy())
                lower, upper = observed_f.confidence_region()
                lower_f.append(lower.detach().cpu().numpy())
                upper_f.append(upper.detach().cpu().numpy())
        return (
            predictions,
            values,
            variance,
            lower_pred,
            upper_pred,
            mean_f,
            lower_f,
            upper_f,
        )

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Exact mll loss")
        plt.title("Loss")
        plt.savefig("loss.png")
        plt.close()

        # plt.plot(self.mse_mean, label="Training mse")

        plt.plot(self.val_losses, label="Validation crps")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("crps")
        plt.title("crps")
        plt.savefig("crps.png")
        plt.close()

