"""Fixed quantities of HWSF (Section IV of the paper and Supplementary Sections S1-S5)."""

# Study prefectures in the order used throughout the code.
PREFECTURES = ("fukui", "ishikawa", "niigata", "toyama")

# Daily inputs cover April 1 to July 30 (121 days); the cutoff is July 31, 00:00 JST.
N_DAYS = 121
# Day ranges of April, May, June, and July within the 121-day window.
MONTHS = ((0, 30), (30, 61), (61, 91), (91, 121))

# Daily weather channels. Precipitation P enters as log(1 + P); VPD is the vapor pressure deficit (kPa).
WEATHER_CHANNELS = (
    "log_precipitation",   # log(1 + P), P in mm per day
    "solar_radiation",     # MJ m-2 day-1
    "relative_humidity",   # %
    "max_temperature",     # deg C
    "mean_temperature",    # deg C
    "min_temperature",     # deg C
    "vpd",                 # kPa
    "radiation_x_vpd",     # solar radiation times VPD
)
# The annual-weather network reads all eight channels; the local-weather network reads the six weather variables.
LOCAL_WEATHER_CHANNELS = (0, 1, 2, 3, 4, 5)

# Six weather summaries (prefecture-year means): (first day, last day, channel index).
WEATHER_SUMMARIES = (
    ("05-21", "06-10", 4),  # temperature, May 21 - June 10
    ("05-21", "06-10", 1),  # solar radiation, May 21 - June 10
    ("07-21", "07-30", 4),  # temperature, July 21-30
    ("07-21", "07-30", 6),  # vapor pressure deficit, July 21-30
    ("07-21", "07-30", 0),  # log precipitation, July 21-30
    ("06-01", "06-20", 1),  # solar radiation, June 1-20
)
# Four CFSv2 forecast fields: temperature and log(1 + precipitation rate) at nominal July leads +1 and +2.
FORECAST_FIELDS = ("temperature_lead1", "temperature_lead2", "log_precipitation_rate_lead1", "log_precipitation_rate_lead2")

# 30 SST-index values: January-June of Nino 1+2, 3, 4, 3.4 (NOAA CPC) and of NINO.WEST (JMA).
SST_INDEX_NAMES = tuple(f"{name}_m{m:02d}" for m in range(1, 7) for name in ("nino12", "nino3", "nino4", "nino34")) + \
    tuple(f"ninowest_m{m:02d}" for m in range(1, 7))

# Equation (6): base prefecture yield.
WEIGHT_ANNUAL = 0.875
WEIGHT_FIXED_FILTER = 0.125
WEIGHT_WEATHER_FORECAST = 0.25
WEIGHT_SST_INDEX = 0.75

# Ensemble sizes (Algorithm 1).
N_WEATHER_MEMBERS = 6
N_CORRECTION_INITIALIZATIONS = 5
N_SST_REPLICAS = 10

# Correction networks bound their scalar output with BOUND * tanh(.).
CORRECTION_BOUND = 5.0
# Scale of the correction losses (9)-(11); it sets only the optimization scale.
LOSS_SCALE = 20.0
# Prediction-state features are divided by this value and clipped.
STATE_SCALE = 20.0
CLIP = 6.0
