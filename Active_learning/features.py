import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from torch.utils.data import TensorDataset, DataLoader

import torch

# from sentence_transformers import SentenceTransformer

# from torch_geometric.data import HeteroData, download_url, extract_zip

# from torch_geometric.transforms import RandomLinkSplit, ToUndirected

 

def get_scaler(scaler):

    scalers = {

        "minmax": MinMaxScaler,

        "standard": StandardScaler,

        "maxabs": MaxAbsScaler,

        "robust": RobustScaler,

    }

    return scalers.get(scaler.lower())()

 

 

def generate_time_lags(df, n_lags, Y):

    df_n = df.copy()
    if n_lags > 0:
        for n in range(1, n_lags + 1):
            df_n[f"lag{n}"] = df_n['RU+DSY'].shift(n)
        df_n = df_n.iloc[n_lags:]

    return df_n

 

 

def feature_label_split(df, target_col):

    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

 

def load_node_csv(path, index_col, encoders=None, **kwargs):

    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

 

# class SequenceEncoder(object):

#     def __init__(self, model_name='all-MiniLM-L6-v2', device=None):

#         self.device = device

#         self.model = SentenceTransformer(model_name, device=device)

 

#     @torch.no_grad()

#     def __call__(self, df):

#         x = self.model.encode(df.values, show_progress_bar=True,

#                               convert_to_tensor=True, device=self.device)

#         return x.cpu()

 

# class GenresEncoder(object):

#     def __init__(self, sep='|'):

#         self.sep = sep

 

#     def __call__(self, df):

#         genres = set(g for col in df.values for g in col.split(self.sep))

#         mapping = {genre: i for i, genre in enumerate(genres)}

 

#         x = torch.zeros(len(df), len(mapping))

#         for i, col in enumerate(df.values):

#             for genre in col.split(self.sep):

#                 x[i, mapping[genre]] = 1

#         return x

 

def train_val_test_split(df, target_col, test_ratio):

    val_ratio = test_ratio / (1 - test_ratio)

    X, y = feature_label_split(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)#shuffle=True

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test

def feature_prep_synthetic(df, batch_size):
    # Xs = ['Type', 'Air temperature [K]',
    #   'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]']
    # Y = 'Tool wear [min]'
    # q25, q75 = np.percentile(y, 25), np.percentile(y, 75)
    # iqr = q75 - q25
    # cut_off = iqr * 1.5
    # lower, upper = q25 - cut_off, q75 + cut_off
    # outliers = [x for x in y if x < lower or x > upper]
    # print('Identified outliers: %d' % len(outliers))
    # # remove outliers
    # df = df[(df[Y] >= lower) & (df[Y] <= upper)].copy()  # [x for x in df if x[Ys[1]] >= lower and x[Ys[1]] <= upper]
    # print('Non-outlier observations: %d' % len(df))
    
    # X = X.ffill()
    # data = data.resample('60Min',  base=30).mean()
    scaler = get_scaler('standard')
    scaler1 = get_scaler('robust')

    input_dim = 0
    df_generated = generate_time_lags(data[Xs + [Y]], input_dim, Y)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_generated, Y, 0.2)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)#shuffle=True

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)
    # y_train = np.log(y_train)
    # y_train = y_train.fillna(0)
    y_train_arr = scaler1.fit_transform(np.array(y_train).reshape(-1, 1))

    y_val_arr = scaler1.transform(np.array(y_val).reshape(-1, 1))
    y_test_arr = scaler1.transform(np.array(y_test).reshape(-1, 1))

    train_features = torch.Tensor(X_train_arr)
    train_targets = torch.Tensor(y_train_arr)

    val_features = torch.Tensor(X_val_arr)
    val_targets = torch.Tensor(y_val_arr)
    test_features = torch.Tensor(X_test_arr)
    test_targets = torch.Tensor(y_test_arr)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    train_loader_one = DataLoader(train, batch_size=1, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    return X_train,y_train, train_features, train_targets, X_test,y_test, scaler,scaler1, train_loader,val_loader, test_loader, train_loader_one, test_loader_one



def feature_prep_air_quality(df, batch_size):
        
    Xs = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
           'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
           'PT08.S5(O3)','RH', 'AH','hour', 'month']
    Y = 'T'
    q25, q75 = np.percentile(df[Y], 25), np.percentile(df[Y], 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    outliers = [x for x in df[Y] if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    # remove outliers
    df = df[(df[Y] >= lower) & (df[Y] <= upper)].copy()  # [x for x in df if x[Ys[1]] >= lower and x[Ys[1]] <= upper]
    print('Non-outlier observations: %d' % len(df))
    data = df.ffill()
    # data = data.resample('60Min',  base=30).mean()
    scaler = get_scaler('standard')
    scaler1 = get_scaler('standard')

    input_dim = 0
    df_generated = generate_time_lags(data[Xs + [Y]], input_dim, Y)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_generated, Y, 0.2)
    # test_df = generate_time_lags(test[Xs + [Y]].ffill(), input_dim)
    # val_df = generate_time_lags(val[Xs + [Y]].ffill(), input_dim)
    
    # X_train = train_df[Xs]
    # y_train = train_df[[Y]]
    # # print(val_df)
    # X_val = val_df[Xs]
    # y_val = val_df[[Y]]
    # X_test = test_df[Xs]
    # y_test = test_df[[Y]]


    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)
    y_train_arr = scaler1.fit_transform(y_train)

    y_val_arr = scaler1.transform(y_val)
    y_test_arr = scaler1.transform(y_test)

    train_features = torch.Tensor(X_train_arr)
    train_targets = torch.Tensor(y_train_arr)

    val_features = torch.Tensor(X_val_arr)
    val_targets = torch.Tensor(y_val_arr)
    test_features = torch.Tensor(X_test_arr)
    test_targets = torch.Tensor(y_test_arr)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    train_loader_one = DataLoader(train, batch_size=1, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    return X_train,y_train, train_features, train_targets, X_test,y_test, scaler,scaler1, train_loader,val_loader, test_loader, train_loader_one, test_loader_one


def feature_prep(df, val, test, batch_size):
        
    Xs = ['draft_aft_telegram', 'draft_fore_telegram', 'stw',
       'diff_speed_overground', 'awind_vcomp_provider', 'awind_ucomp_provider',
       'rcurrent_vcomp', 'rcurrent_ucomp', 'comb_wind_swell_wave_height',
       'timeSinceDryDock']

    Y = 'power'
    q25, q75 = np.percentile(df[Y], 25), np.percentile(df[Y], 75)

    iqr = q75 - q25

    cut_off = iqr * 1.5

    lower, upper = q25 - cut_off, q75 + cut_off

    outliers = [x for x in df[Y] if x < lower or x > upper]

    print('Identified outliers: %d' % len(outliers))

    # remove outliers

    df = df[(df[Y] >= lower) & (

                df[Y] <= upper)].copy()  # [x for x in df if x[Ys[1]] >= lower and x[Ys[1]] <= upper]

    print('Non-outlier observations: %d' % len(df))

 

    # tdi = pd.DatetimeIndex(df.index_date)

    # df.set_index(tdi, inplace=True)

    # df.drop_duplicates(subset="index_date", keep='last', inplace=True)

    # data = df.asfreq('min')

    # data.drop(columns=['index_date', 'index_num'], inplace=True)

    data = df.ffill()

    # data = data.resample('60Min',  base=30).mean()

    scaler = get_scaler('standard')

    scaler1 = get_scaler('standard')

    input_dim = 0
    train_df = generate_time_lags(data[Xs + [Y]], input_dim,Y)
    #X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_generated, Y, 0.2)
    test_df = generate_time_lags(test[Xs + [Y]].ffill(), input_dim, Y)
    val_df = generate_time_lags(val[Xs + [Y]].ffill(), input_dim, Y)
    
    X_train = train_df[Xs]
    y_train = train_df[[Y]]
    # print(val_df)
    X_val = val_df[Xs]
    y_val = val_df[[Y]]
    X_test = test_df[Xs]
    y_test = test_df[[Y]]


    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)
    y_train_arr = scaler1.fit_transform(y_train)

    y_val_arr = scaler1.transform(y_val)

    y_test_arr = scaler1.transform(y_test)

 

    train_features = torch.Tensor(X_train_arr)
    train_targets = torch.Tensor(y_train_arr)

    val_features = torch.Tensor(X_val_arr)

    val_targets = torch.Tensor(y_val_arr)

    test_features = torch.Tensor(X_test_arr)

    test_targets = torch.Tensor(y_test_arr)

 

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

 

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)

    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

    train_loader_one = DataLoader(train, batch_size=1, shuffle=False, drop_last=True)

    #val_loader_one = DataLoader(val, batch_size=1, shuffle=False, drop_last=True)

    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    return X_train,y_train, train_features, train_targets, X_test,y_test, scaler,scaler1, train_loader,val_loader, test_loader, train_loader_one, test_loader_one


def inverse_transform(scaler, df, columns):

    for col in columns:

        df[col] = scaler.inverse_transform(df[col])

    return df

 

 

def format_predictions(predictions, values, variance, lower_pred, upper_pred, mean_f, lower_f, upper_f, scaler):

    # vals = np.concatenate(values, axis=0).ravel()

    variance = np.concatenate(variance, axis=0).ravel()

    pred_likelihood = np.concatenate(predictions, axis=0).ravel()

    lower_likelihood = np.concatenate(lower_pred, axis=0).ravel()

    upper_likelihood = np.concatenate(upper_pred, axis=0).ravel()

    mean_pred = np.concatenate(mean_f, axis=0).ravel()

    lower_pred = np.concatenate(lower_f, axis=0).ravel()

    upper_pred = np.concatenate(upper_f, axis=0).ravel()

    df_result = pd.DataFrame()

    # scale = (y_train_max - y_train_min) + y_train_min

    df_result['actual'] = values#*scale

    df_result['prediction'] = pred_likelihood#*scale

    df_result['variance'] = variance#*scale

    df_result['lower_likelihood'] = lower_likelihood#*scale

    df_result['upper_likelihood'] = upper_likelihood#*scale

    df_result['mean_pred'] = mean_pred#*scale

    df_result['lower_pred'] = lower_pred#*scale

    df_result['upper_pred'] = upper_pred#*scale

    # df_result = pd.DataFrame(data={"actual": vals, "prediction": pred_likelihood, "variance":variance,"lower_likelihood":lower_likelihood, "upper_likelihood":upper_likelihood,

    # "mean_pred":mean_pred, "lower_pred":lower_pred, "upper_pred":upper_pred}, index=df_test.head(len(vals)).index)

    df_result = df_result.sort_index()

    unscaled = df_result.copy()

    df_result = inverse_transform(scaler, df_result, [["actual", "prediction", "variance", "lower_likelihood", "upper_likelihood", "mean_pred", "lower_pred", "upper_pred"]])

    return df_result

#

# def eval_crps(pred, truth):

#     """

#     Evaluate continuous ranked probability score, averaged over all data

#     elements.

#

#     **References**

#

#     [1] Tilmann Gneiting, Adrian E. Raftery (2007)

#         `Strictly Proper Scoring Rules, Prediction, and Estimation`

#         https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

#

#     :param torch.Tensor pred: Forecasted samples.

#     :param torch.Tensor truth: Ground truth.

#     :rtype: float

#     """

#     opts = dict(device=pred.device, dtype=pred.dtype)

#     num_samples = pred.size(0)

#     pred = pred.sort(dim=0).values

#     diff = pred[1:] - pred[:-1]

#     weight = (torch.arange(1, num_samples, **opts) *

#               torch.arange(num_samples - 1, 0, -1, **opts))

#     weight = weight.reshape(weight.shape + (1,) * truth.dim())

#

#     return ((pred - truth).abs().mean(0) - (diff * weight).sum(0) / num_samples ** 2).mean().item()

 

 

def calculate_metrics(df):

    return {'mae' : mean_absolute_error(df.actual, df.prediction),

            'mse' : mean_squared_error(df.actual, df.prediction) ** 0.5,

            'r2' : r2_score(df.actual, df.prediction),

            }


