import numpy as np

def get_idle_mask(series, window_size, threshold):
    if window_size % 2 == 0:
        window_size += 1
    
    n, d = series.shape    
    idle = np.full(n, False)
    
    half_window = window_size // 2
    for center in range(half_window, n-half_window):
        window_start = center-half_window
        window_end   = center+half_window+1
        if np.all(np.var(series[window_start:window_end, :], axis=0) < threshold):
            # idle[center] = True
            idle[window_start:window_end] = True
    return idle


def get_segments(mask):
    segments = np.argwhere(np.diff(mask, prepend=False, append=False))
    segments = segments.reshape(len(segments) // 2, 2)
    return segments


def get_idle_segments(series, window_size, threshold):
    idle_mask = get_idle_mask(series, window_size, threshold)
    segments = get_segments(idle_mask)
    return list(segments)


def get_start_mask(series, idle_mask, quantile):
    # mean of the idle values
    idle_mean  = np.mean(series[idle_mask, :], axis=0)
    # determine the distance threshold
    distances          = np.sum((series[~idle_mask] - idle_mean) ** 2, axis=1)
    distance_threshold = np.quantile(distances, quantile)
    # construct the start mask
    start_mask = (np.sum((series - idle_mean) ** 2, axis=1) <= distance_threshold)
    start_mask[idle_mask] = False
    return start_mask.astype(bool)