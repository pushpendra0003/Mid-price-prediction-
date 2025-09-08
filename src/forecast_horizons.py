# Forecast horizon utilities

def get_forecast_horizons(horizons=None):
    # Default: predict 1, 5, and 10 steps ahead
    if horizons is None:
        horizons = [1, 5, 10]
    return horizons
