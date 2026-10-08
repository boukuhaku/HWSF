# Data

HWSF estimates the municipal rice yield of a target year *t* from information available before the cutoff of
July 31, 00:00 JST. Apart from the MAFF yields of the evaluation records in `results/`, the source data are not
redistributed in this repository; this page lists them and describes how `hwsf.prepare` turns them into model inputs.

## Study area

80 municipalities in Fukui (17), Ishikawa (19), Niigata (29), and Toyama (15). Every array follows the *model
order*: prefectures alphabetically, then municipalities by area code. `HWSF.municipalities` lists the order of a
checkpoint as (prefecture, area code) pairs.

## Sources

| Input | Provider | Notes |
|---|---|---|
| Municipal rice yields (kg/10 a) | MAFF crop statistics (e-Stat) | Reference values; brown rice after a 1.70-mm sieve. Used from 2008 for training, plus the three years before *t* as model input. |
| Prefecture rice yields | MAFF crop statistics | Only for fitting the SST-index correction. |
| Daily weather | NARO 1-km agro-meteorological grid data | Registration required. Six variables at one representative grid cell per municipality, April 1 – July 30. |
| SST maps | NOAA OISST v2.1 monthly means (PSL THREDDS) | January–June, sampled every 2° on 50 × 180 points (38.875° S – 59.125° N). |
| SST indices | NOAA CPC `sstoi.indices`; JMA NINO.WEST | Absolute monthly SST (not anomalies), January–June. CPC reports 0.01 °C and JMA 0.1 °C. |
| Seasonal forecasts | NCEP CFSv2 ensemble mean, July 8 runs | 2-m temperature and precipitation rate at nominal leads +1 and +2 (from 2012). |
| Prefecture weather | NASA POWER (AG community) | Monthly April–July means, only for fitting the SST-index correction. |
| Administrative boundaries | MLIT National Land Numerical Information N03 | For choosing the representative grid cell. |

### Representative grid cell

For each municipality, the representative cell is the 1-km cell nearest to the paddy-weighted centroid of the
municipality (NARO land use 2009, MLIT boundaries 2014). The same cell is used for all years, for the regional
inputs, and for interpolating the forecasts.

## Model inputs

`hwsf.data.TargetYearInputs` holds the inputs of one target year (`load_inputs` / `save_inputs` read and write them as
`.npz` files):

| Field | Shape | Content |
|---|---|---|
| `municipality_prefecture` | (n,) | prefecture of each municipality, model order |
| `regional_inputs` | (n, 26) | 24 April–July monthly climate means of the six daily variables over the training years (index 6·month + variable), latitude, longitude |
| `weather` | (n, 121, 8) | daily channels of year *t*, April 1 – July 30 |
| `reference_weather` | (n, 3, 121, 8) | daily channels of *t* − 3, *t* − 2, *t* − 1 |
| `reference_yields` | (n, 3) | observed yields of *t* − 3, *t* − 2, *t* − 1 (kg/10 a) |
| `forecasts` | (4, 4) | prefecture means of temperature (°C) and log(1 + precipitation rate in mm/day) at leads +1 and +2 |
| `sst_fields` | (6, 50, 180) | January–June OISST (°C) on the 2° sample grid, NaN on land |
| `sst_indices` | (30,) | Niño 1+2, 3, 4, 3.4 for January–June (month-major), then NINO.WEST January–June |

### Daily channels

The eight channels are built from the six NARO variables (`hwsf.prepare.naro_channels`):

| Channel | Content | Unit |
|---|---|---|
| 0 | log(1 + precipitation) | log(1 + mm) |
| 1 | solar radiation | MJ m⁻² day⁻¹ |
| 2 | relative humidity | % |
| 3 | maximum temperature | °C |
| 4 | mean temperature | °C |
| 5 | minimum temperature | °C |
| 6 | vapor pressure deficit, (e_s(T_max) + e_s(T_min))/2 · (1 − RH/100) | kPa |
| 7 | solar radiation × vapor pressure deficit | |

`read_naro_daily` drops all days outside April 1 – July 30 before the values are used and raises the daily maximum
temperature to the daily mean where the mean exceeds it (one record in the study data). The model standardizes the
channels with municipality-by-calendar-day means and pooled channel SDs of the training years, which are stored in each
checkpoint.

### Weather summaries and forecasts

The six weather summaries (temperature and solar radiation during May 21 – June 10, solar radiation during
June 1–20, and temperature, vapor pressure deficit, and log precipitation during July 21–30) are computed inside the
model from `weather`. The forecasts are interpolated bilinearly from the 1° CFSv2 grid to the representative cells,
converted, and averaged per prefecture (`hwsf.prepare.forecast_fields`).

### SST maps

`hwsf.prepare.fetch_oisst_fields(year)` downloads the six monthly fields of a year from the PSL OPeNDAP server. Inside
the model, the fields are averaged over 10° × 10° cells (cos-latitude weights), standardized with the 1982–2011
monthly climatology and SD (floor 0.2 °C), and clipped to [−6, 6]; the ocean fraction of each cell forms the seventh
channel and a zero mask indicator the eighth.

## Uncertainty sources

`hwsf.uncertainty.load_sources` reads the input-uncertainty sources of a target year from an `.npz` file:

| Field | Shape | Content |
|---|---|---|
| `representativeness` | (n, 121, 6) | paddy-area-weighted mean of all NARO cells of the municipality minus the value at the representative cell, for mean, maximum, and minimum temperature, precipitation, solar radiation, and relative humidity |
| `sst_error_sd` | (6, 50, 180) | monthly mean of the daily OISST error SDs (°C), 0 on land |
| `index_half_width` | (30,) | optional; default 0.005 °C for the CPC indices and 0.05 °C for NINO.WEST |
| `yield_half_width` | () | optional; default 0.5 kg/10 a |

## Time boundary

All inputs of year *t* are observed before July 31, 00:00 JST: the daily weather ends on July 30, the SST maps and
indices end in June, and the forecasts come from the July 8 runs. Fitted quantities (standardization statistics,
regressions, networks) use only records of years before *t*.
