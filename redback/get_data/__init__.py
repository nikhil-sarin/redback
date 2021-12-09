# from redback.get_data import directory, open_data, swift, utils
import redback.get_data.directory
import redback.get_data.open_data
import redback.get_data.swift
import redback.get_data.utils

def get_swift_data(grb, transient_type, data_mode='flux', bin_size=None):
    getter = swift.SwiftDataGetter(grb=grb, transient_type=transient_type, data_mode=data_mode, bin_size=bin_size)
    getter.get_data()
    return getter
