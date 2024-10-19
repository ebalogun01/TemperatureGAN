import json
import os
import numpy as np
from train import Trainer


USE_NUM_TRAINING_SAMPLES = 1000


def main(interactive=False):
    # OPTIONAL CODE BLOCK
    if interactive:
        config_path = input("Config file path: ")
        assert type(config_path) is str # might be extraneous
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        curr_working_dir = os.getcwd()
        for root, dirs, files, in os.walk(curr_working_dir):
            for file in files:
                if file == "config.json":
                    with open(file, "r") as f:
                        config = json.load(f)

    print("Beginning training pre-load...")
    if config['gmst']['use']:
        from gan import GeneratorGMST as Generator
        from gan import DiscriminatorGMST as Discriminator
        from gan import DiscriminatorTGMST as DiscriminatorT
    else:
        from gan import Generator, Discriminator, DiscriminatorT

    # init critic
    D_T = DiscriminatorT()
    D_S = Discriminator()
    D = [D_S, D_T]

    # init Generator
    G = [Generator(config)]

    data = np.load(config["data_file"][0])[:USE_NUM_TRAINING_SAMPLES]
    loc_label = np.load(config["data_file"][3])[:USE_NUM_TRAINING_SAMPLES]
    month_label = np.load(config["data_file"][1])[:USE_NUM_TRAINING_SAMPLES]
    period_label = np.load(config["data_file"][2])[:USE_NUM_TRAINING_SAMPLES]

    if config['gmst']['use']:
        gmst_data = np.load(config['gmst_data'])
        if config['gmst']['normalize'] and not config['gmst']['precalc']:
            config['gmst']['mean'] = gmst_data.mean()
            config['gmst']['std'] = gmst_data.std()
            config['gmst']['max'] = gmst_data.max()
            config['gmst']['min'] = gmst_data.min()

            # Normalize to between 0 and 1
            gmst_data = (gmst_data - gmst_data.min()) / (gmst_data.max() - gmst_data.min())

            # Standardize to have mean 0 and std 1
            gmst_data = (gmst_data - gmst_data.mean()) / gmst_data.std()

        labels = np.concatenate((loc_label, month_label, period_label, gmst_data), axis=1)
        gmst_data = None    # free up memory
        assert labels.shape[1] == 19
    elif config['co2']['use']:
        co2_data = np.load(config['co2_data'])
        if config['co2']['normalize'] and not config['co2']['precalc']:
            config['co2']['mean'] = co2_data.mean()
            config['co2']['std'] = co2_data.std()
            config['co2']['max'] = co2_data.max()
            config['co2']['min'] = co2_data.min()

            # Normalize to between 0 and 1
            co2_data = (co2_data - co2_data.min()) / (co2_data.max() - co2_data.min())

            # Standardize to have mean 0 and std 1
            co2_data = (co2_data - co2_data.mean()) / co2_data.std()

        labels = np.concatenate((loc_label, month_label, period_label, co2_data), axis=1)
        co2_data = None
        assert labels.shape[1] == 19
    else:
        labels = np.concatenate((loc_label, month_label, period_label), axis=1)
        assert labels.shape[1] == 15
    labels_1, labels_2, labels_3 = None, None, None    # clear memory

    trainer = Trainer(config, D, G, data, labels)
    print("Pre-load complete. Beginning training...")

    trainer.train()


if __name__ == "__main__":
    main()
