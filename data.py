"""
Data Preprocessing for the thermal images,
"""
from dataclasses import dataclass
import torch
import numpy as np
import torch.nn.functional as F
import json

@dataclass
class TrainDataLoader:
    config: dict
    temporal_len: int = 24 # hours
    start_lon: int = 1  # grid boxes
    start_lat: int = 1  # grid boxes count

    def __post_init__(self):
        self.start_month = self.config["start_month"]
        self.end_month = self.config["end_month"]
        self.start_lat = self.config["start_loc_lat"]
        self.start_lon = self.config["start_loc_lon"]
        self.num_lon_pts = self.config["num_lon_pts"]  # stopping at west coast regions (CA, OR, NV, AZ(partly), ID(partly))
        self.num_lat_pts = self.config["num_lat_pts"]
        self.locations = [(lon, lat) for lon in range(self.start_lon, self.num_lon_pts + 1) for lat in
                     range(self.start_lat, self.num_lat_pts + 1)]
        self.examples = 10

    def crop_data_by_loc(self):
        pass

    def sample_data_by_month(self, month, period):
        """This function collates all training data for a month within given time period"""

    def label_data(self):
        locations = [(lon, lat) for lon in range(self.start_lon, self.num_lon_pts + 1) for lat in
                     range(self.start_lat, self.num_lat_pts + 1)]
        spatial_label = torch.cat(())

    def load_dataset(self, period, noise_dist="gaussian", save_labels=True):
        """This will prepare dataset for an entire period..."""
        #TODO: include variable to allow parallel loading of these datasets
        base_path_str = "data/CA_train/data_CA_OR_WA_period_{}/train_{}_1-12loc_{}_{}.npy"
        initial_dataset = True
        for month in range(self.start_month, self.end_month+1):
            month_label = month - 1
            loc_labels_cum = np.array([])
            month_train_label = F.one_hot(torch.tensor([month_label]), num_classes=12).float()
            for loc in self.locations:
                # loc is a tuple here
                spatial_label = np.array(list(loc))
                loc_data = np.load(base_path_str.format(period, period, loc[0], loc[1]), allow_pickle=True)
                loc_month_data = loc_data[month_label]
                num_data_pts = len(loc_month_data)
                if initial_dataset:
                    month_labels = np.tile(month_train_label, (num_data_pts, 1))
                    thermal_images = loc_month_data
                    loc_labels = np.tile(spatial_label, (num_data_pts, 1))
                    initial_dataset = False
                else:
                    month_labels = np.append(month_labels, np.tile(month_train_label, (num_data_pts, 1)), axis=0)
                    thermal_images.extend(loc_month_data[0:])
                    loc_labels = np.append(loc_labels, np.tile(spatial_label, (num_data_pts, 1)), axis=0)
        if save_labels:
            np.save("training_data_month_labels_" + period, month_labels)
            np.save("training_data_loc_labels_" + period, loc_labels)
            np.save("training_data_temperature_" + period, thermal_images)
        #
        # return loc_labels, month_labels, thermal_images

    def load_cum_dataset(self):
        for month in range(self.start_month, self.end_month):
            month_idx = month - 1
            self.load_dataset(month_idx)

def test():
    with open('data.json', 'r') as fp:
        config = json.load(fp)
    periods = ["2003-2006"]
    loader = TrainDataLoader(config)
    for period in periods:
        loader.load_dataset(period)


if __name__ == "__main__":
    #   testing the dataloader that eventually was used in the test folder
    test()



