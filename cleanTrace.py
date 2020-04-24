#!/usr/bin/env python3

import csv
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], skiprows=[4], usecols=['Start', 'Duration', 'Size', 'Name'], dtype={'Start':float, 'Duration':float, 'Size':float, 'Name':str}, comment="=")
