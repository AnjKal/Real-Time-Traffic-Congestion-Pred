"""Readers for the Bangalore feature HDF5 files across pandas versions."""
import h5py
import numpy as np
import pandas as pd


def read_feature_tables(path):
    """Read pandas HDF5 tables without relying on PyTables datetime decoding."""
    tables = {}
    with h5py.File(path, "r") as store:
        for key in store.keys():
            group = store[key]
            columns = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in group["axis0"][:]
            ]
            timestamps = pd.to_datetime(
                np.asarray(group["axis1"][:], dtype=np.int64), unit="us"
            )
            values = np.asarray(group["block0_values"][:], dtype=np.float64)
            tables[key] = pd.DataFrame(values, index=timestamps, columns=columns)
    return tables
