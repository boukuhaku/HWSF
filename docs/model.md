# Model

This page maps the components of HWSF in the paper (Section IV, Figs. 6 and 7, Supplementary Table S1) to the code.

## Hierarchy

The observed yield of municipality *i* in prefecture *c* and year *t* is decomposed into the equal-weight
prefecture-year mean and the municipal deviation, $y_{ict}=\mu_{ct}+d_{ict}$. HWSF predicts both components with
separate branches, and the predicted deviations are centered within each prefecture-year:

$$\hat y_{ict}=\hat\mu_{ct}+\hat d_{ict},\qquad \tfrac{1}{n_c}\textstyle\sum_i\hat d_{ict}=0 .$$

## Networks

| Paper | Code | Configuration | Parameters per network |
|---|---|---|---:|
| Regional network | `nn.networks.RegionalNetwork` | dense 26 → 16 → 16 → 1, tanh | 721 |
| Annual-weather network | `WeatherNetwork(in_channels=10, depthwise=True)` | 1×1 projection to 12 channels; three kernel-7 depthwise residual blocks (dilation 1, 2, 4); monthly means (48) + regional features (16); dense 64 → 16 → 1, GELU | 1,945 |
| Local-weather network with FiLM | `WeatherNetwork(in_channels=8, depthwise=False, film=True)` | as above with full-channel convolutions; after each block, dense 16 → 24 gives scale and offset terms bounded by 0.5 tanh(·) | 5,917 |
| History weighting network | `HistoryNetwork` | dense 34 → 16 → 1, GELU; scores → tanh → softmax over the three reference years | 577 |
| History correction network | `HistoryNetwork` | dense 34 → 16 → 1, GELU; mean of the three outputs | 577 |
| Prefecture correction network | `ResidualMLP(294)` | dense 294 → 64 → 64 → 1, GELU, hidden residual connection, 5 tanh(·) output | 23,105 |
| Municipal correction network | `ResidualMLP(147)` | dense 147 → 64 → 64 → 1, as above; output centered | 13,697 |
| SST encoder | `SSTEncoder` | two 3×3 convolutions with 8 channels (circular padding in longitude, edge padding in latitude), 2×2 and 1×3 average pooling, residual second convolution, dense 240 → 16, GELU | 5,024 |
| SST decoder (pretraining only) | `SSTDecoder` | dense 16 → 64 → 2160, reshaped to 6 × 10 × 36 | – |
| SST fusion network | `ResidualMLP(80)` | prefecture hidden representation (64) + SST-map features (16); dense 80 → 64 → 64 → 1, as above | 9,409 |

The deployed model of one target year averages six weather members (`HWSF.members`). Each member has one regional,
annual-weather, local-weather, history weighting, and history correction network, five prefecture and five municipal
correction networks, and ten SST fusion networks; the ten SST encoders are shared. In total the model has 1,777,262
network parameters.

## Statistical calibration

| Paper | Code |
|---|---|
| fixed-filter prediction $p^{\mathrm{fil}}_{ct}$: six banks of 768 fixed random temporal filters (seeds 8701–8706) on prefecture-mean standardized weather; monthly maxima and positive proportions (6208 values) plus the 16 prefecture-mean regional features; ridge regression | `calibration.FixedFilterRegression`, `calibration.make_filter_bank` |
| weather-and-forecast correction $c^{\mathrm{wf}}_{ct}$: six weather summaries and four forecast fields, prefecture-centered; ridge regression | `calibration.WeatherForecastCorrection` |
| SST-index correction $c^{\mathrm{SST}}_{t}$: 30 SST-index values; ridge regression fitted to official prefecture yields with 20 NASA POWER weather means that are used only during fitting | `calibration.SSTIndexCorrection` |
| base prefecture yield $\mu^0_{ct}=0.875p^{\mathrm{ann}}_{ct}+0.125p^{\mathrm{fil}}_{ct}+0.25c^{\mathrm{wf}}_{ct}+0.75c^{\mathrm{SST}}_{t}$ | `HWSF.predict` |

## Municipal features

Each municipality is described by 147 standardized features (Table II of the paper):

| Group | Count |
|---|---:|
| annual-weather and regional representations | 16 + 16 |
| weather summaries, forecasts, and SST-index values | 6 + 4 + 30 |
| prediction-state features | 8 |
| current local-weather representation | 16 |
| three historical representations and residuals | 3 × (16 + 1) |

The prefecture correction network reads the mean and population SD of these features within the prefecture-year
(294 values). The eight prediction-state features describe the base prefecture yield relative to its history, the
weighted statistical corrections, and the spread among the six weather members (Supplementary Section S5).

## Outputs

`HWSF.predict(inputs, return_components=True)` returns

| Key | Paper |
|---|---|
| `estimate` | $\hat y_{ict}$ (mean over the six members) |
| `prefecture_level` | $\hat\mu_{ct}$ |
| `municipal_deviation` | $\hat d_{ict}$ |
| `base_prefecture_yield` | $\mu^0_{ct}$ of each member |
| `annual_weather_prediction` | $p^{\mathrm{ann}}_{ct}$ |
| `fixed_filter_prediction` | $p^{\mathrm{fil}}_{ct}$ |
| `weather_forecast_correction` | $c^{\mathrm{wf}}_{ct}$ |
| `sst_index_correction` | $c^{\mathrm{SST}}_{t}$ |
| `base_municipal_deviation` | $d^0_{ict}$ of each member |
| `sst_features` | standardized SST-map features of the ten encoders |

## Numerical precision

Weights are stored in float32 (the regional network in float64). `HWSF.from_checkpoint` runs inference in float64
and reproduces the published estimates of the 560 evaluation records to within 1e-5 kg/10 a.
