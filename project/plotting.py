import matplotlib.pyplot as plt
import cartopy
import structs
import numpy as np
from ttu_tower import constants as tower_constants

def time_profile(tower_data: structs.TowerData, variable: str, start_time: str|None=None, end_time: str|None=None):
    # Plot the tower profile of a variable (e.g. ws, theta) in a heat-map format over time, between start_time and end_time
    df_full = tower_data.profiles[variable]
    if start_time or end_time and not (start_time and end_time):
        raise Exception("Both start and end time must be specified, or neither.")
    df = df_full[(df_full.index >= start_time) & (df_full.index <= end_time)] if start_time is not None else df_full
    fig, ax = plt.subplots(figsize=(12, 6))
    c = ax.pcolormesh(df.index, df.columns, df.T, shading="auto", cmap="viridis")
    ax.set_yticks(df.columns)
    ax.set_yticklabels([str(tower_constants.HEIGHTS[int(b)]) for b in df.columns])
    ax.set_title(f"Tower Profile ({tower_data.source}) of {variable}" + ("from {start_time} to {end_time}" if start_time is not None else ""))
    ax.set_xlabel("Time")
    ax.set_ylabel("Height (m)")
    fig.colorbar(c, ax=ax, label=structs.MESONET_UNITS.get(variable, ""))
    plt.show()
    return
