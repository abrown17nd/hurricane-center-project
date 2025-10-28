#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:08:21 2024

@author: elliott
"""

import xarray as xr 
import matplotlib.pyplot as plt
import numpy as np 
from datetime import datetime 
from pyproj import Proj 
import scipy 


from datetime import datetime, timedelta

def generate_intervals(start_time, end_time, interval):
    """Generates a list of 10-minute intervals between two datetime objects."""

    current_time = start_time
    intervals = []

    while current_time < end_time:
        intervals.append(current_time.strftime('%H%M'))
        current_time += interval

    return intervals

def generate_intervals_datetime(start_time, end_time, interval):
    """Generates a list of 10-minute intervals between two datetime objects."""

    current_time = start_time
    intervals = []

    while current_time < end_time:
        intervals.append(current_time)
        current_time += interval

    return intervals

# Example usage
start_time = datetime(2024, 11, 25, 10, 0)  # 10:00 AM
end_time = datetime(2024, 11, 25, 12, 0)    # 12:00 PM
interval = timedelta(minutes=10)

intervals = generate_intervals(start_time, end_time, interval)
