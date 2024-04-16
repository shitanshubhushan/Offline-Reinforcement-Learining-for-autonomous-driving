#import pandas as pd
#import numpy as np

# get current motion data
class motionState:
    def __init__(self, time_stamp_ms):
        assert isinstance(time_stamp_ms, int)
        self.time_stamp_ms = time_stamp_ms
        self.agent_type = None
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None
        self.psi_rad = None
        self.length = None
        self.width = None
        self.unique_id = None

    def __str__(self):
        return "motionState: " + str(self.__dict__)

    def get_dict_type_data(self):
        return {'agent_type':self.agent_type, 'x':self.x, 'y':self.y, 'vx':self.vx, 'vy':self.vy,'psi_rad':self.psi_rad,
                'length': self.length, 'width':self.width, 'unique_id': self.unique_id, 'agent_type': self.agent_type}

# each unique_id (= case_id * 100 + track_id) represents either a car, pedestrian, or bike
class uniqueTrack:
    def __init__(self, unique_id):
        self.case_id = None
        self.track_id = None
        self.unique_id = unique_id
        self.agent_type = None
        self.length = None
        self.width = None
        self.time_stamp_ms_first = None
        self.time_stamp_ms_last = None
        self.motionState = {}

    def __str__(self):
        string = f"uniqueTrack: case_id={self.case_id}, track_id={self.track_id}, \
                unique_id={self.unique_id}, agent_type={self.agent_type}, \
                length={self.length}, width={self.width}" 
        for key, value in sorted(self.motionState.items()):
            string += "\n    " + str(key) + ": " + str(value)
        return string

# get each uniqueTrack data
def read_uniqueTracks(df):
    uniqueTracks = {}

    for index, row in df.iterrows():
        unique_id = row['unique_id']
        if unique_id not in uniqueTracks:
            track = uniqueTrack(unique_id)
            track.case_id = row['case_id']
            track.track_id = row['track_id']
            track.agent_type = row['agent_type']
            track.length = row['length']
            track.width = row['width']
            track.motionState = {}
            uniqueTracks[unique_id] = track
        else:
            track = uniqueTracks[unique_id]

        timestamp = row['timestamp_ms']
        motion_state = {
            'agent_type': row['agent_type'],
            'x': row['x'], 
            'y': row['y'], 
            'vx': row['vx'], 
            'vy': row['vy'], 
            'psi_rad': row['psi_rad'],
            'length': row['length'],
            'width':row['width'],
            'unique_id': row['unique_id'],
            'agent_type': row['agent_type']
        }

        track.motionState[timestamp] = motion_state

    return uniqueTracks




