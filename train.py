import torch
import torch.nn as nn
import torch.optim as optim
from costfns import wasserstein_gen, wasserstein_critic, mode_seeking
import torch.utils.data as data_utils
import json
import pathlib
import logging
import os
import wandb
import timeit
import datetime


wandb.login()
logging.basicConfig()  # to be done in training (on each training session)


class Trainer:
    # TODO: need to fix image normalization scheme to deal with out-of-distribution liers (regions with no data)
    def __init__(self, config, D, G, D_data, labels, normalize_data=True,
                 train_G=True, data_std=10.88, data_mean=284.1, vegetation_data=None,
                 log_gradients=False, legacy=False):
        self.config = config
        self.D_data = D_data
        self.labels = labels
        self.D = D
        self.G = G
        self.normalize_data = normalize_data
        self.train_G = train_G
        self.data_std = data_std
        self.data_mean = data_mean
        self.num_G_steps = self.config["g_steps"]
        self.num_D_steps = self.config["d_steps"]
        self.norm_mean = self.config["norm_mean"]
        self.norm_std = self.config["norm_std"]
        self.data_min = 0
        self.log_gradients = log_gradients
        # self.data_min = 223.36  # ONLY USE THIS WHEN MEAN PADDING = TRUE, ELSE USE ABOVE
        self.data_max = 322.9
        # self.data_max = 324.35
        self.legacy = legacy

        if self.config["resume"]["status"]:
            if self.legacy:
                c = self.config
                models = torch.load(
                    os.path.join(
                        c["results_dir"], "models", "model_{}".format(c["resume"]["resume_from"])
                    )
                )
                self.D_S = models["Critic"][0]
                self.D_T = models["Critic"][1]
                self.G = models["Generator"]
            else:
                c = self.config
                models = torch.load(
                    os.path.join(
                        c["results_dir"], "models", "model_{}".format(c["resume"]["resume_from"])
                    )
                )
                self.D_S.load_state_dict(models["Critic"][0])
                self.D_T.load_state_dict(models["Critic"][1])
                self.G.load_state_dict(models["Generator"])
        else:
            self.D_T = self.D[1]
            self.D_S = self.D[0]
            self.G = self.G[0]  # should only contain one object

    def train(self):
        with wandb.init(config=self.config, project='legacyFinetuning', notes='testing new finetuning code'):
            c = self.config
            D_data = self.D_data    # modified code to fit wandb
            labels = self.labels

            # Create results directories
            pathlib.Path(c["results_dir"]).mkdir(parents=True, exist_ok=True)
            pathlib.Path(c["results_dir"], "models").mkdir(parents=True, exist_ok=True)
            pathlib.Path(c["results_dir"], "plots").mkdir(parents=True, exist_ok=True)

            # Make Logging file
            PID = os.getpid()
            logging_path = os.path.join(c["results_dir"], "output.out")
            formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s', '%m/%d/%Y %I:%M:%S')
            hdlr1 = logging.FileHandler(logging_path)
            hdlr1.setFormatter(formatter)
            logger = logging.getLogger(str(PID))
            logger.handlers = []
            logger.addHandler(hdlr1)
            logger.setLevel(logging.INFO)

            logging.info("Begin training...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if str(device) == 'cuda':
                devices = torch.cuda.device_count()
                print("GPU count is {}".format(devices))
                # set models in devices
                if int(devices) > 1 and c['parallel']:
                    self.D_T = nn.DataParallel(self.D_T)
                    self.D_S = nn.DataParallel(self.D_S)
                    self.G = nn.DataParallel(self.G)  # shou
                if int(devices) > 1 and not c['parallel']:
                    print("Error: using multi-gpu without assigning parallel")

            logger.info("Training on {}".format(device))
            default_dtype = torch.FloatTensor
            torch.set_default_tensor_type(default_dtype)

            if c["mean_padding"]:
                D_data[D_data == 0] = self.data_mean
                self.data_min = D_data.min()    # corrected 7/18/22

            # Normalize data OR standardize data - OPTIONAL
            # todo: Make changes here later to only do standardization.
            if self.normalize_data:
                # Currently uses both normalization and standardization to constrain the data to range [-1, 1]
                D_data = (D_data - self.data_min) / (self.data_max - self.data_min)
                D_data = (D_data - self.norm_mean) / self.norm_std
            # Send data to tensors (if the entire dataset can fit in RAM, send it to device in the next line)
            D_train_tensor = torch.from_numpy(D_data).type(default_dtype).to(device)
            labels_tensor = torch.from_numpy(labels).type(default_dtype).to(device)

            D_dataset = data_utils.TensorDataset(D_train_tensor, labels_tensor)  # THIS WILL TAKE IN LABELS
            D_dl = data_utils.DataLoader(D_dataset, batch_size=c["samples_per_batch"], shuffle=True)

            if c["resume"]["status"]:
                models = torch.load(
                    os.path.join(c["results_dir"], "models", "model_{}".format(c["resume"]["resume_from"])))
                # assign Gen and Critic models already done in __post__init()
                epoch_start = models["epoch_num"]
                total_batch_tally = models["batch_num"]
                logger.info(f'Resuming training for {c["results_dir"]}')
            else:
                epoch_start = 1
                total_batch_tally = 1
                logger.info("Starting new training for {}".format(c["results_dir"]))

            # Send models to device
            self.G.to(device)
            self.D_T.to(device)
            self.D_S.to(device)

            # Set the Optimizer and scale G LR
            if c["optimizer"] == 'adam':
                if c["resume"]["status"]:
                    logger.info(f'Loading and resuming ADAM optimizer state dict from {c["resume"]["resume_from"]}')
                    gen_optimizer = optim.Adam(self.G.parameters(), lr=c["g_learning_rate"], betas=(c["adam_beta1"],
                                                                                                    c["adam_beta2"]),
                                               weight_decay=c["weight_decay"], amsgrad=False)
                    gen_optimizer.load_state_dict(models['G_optimizer'])
                    critic_optimizer_T = optim.Adam(self.D_T.parameters(), lr=c["d_learning_rate"],
                                                    betas=(c["adam_beta1"],
                                                           c["adam_beta2"]),
                                                    weight_decay=c["weight_decay"], amsgrad=False)
                    critic_optimizer_T.load_state_dict(models['Critic_optimizer_T'])
                    critic_optimizer_S = optim.Adam(self.D_S.parameters(), lr=c["d_learning_rate"],
                                                    betas=(c["adam_beta1"],
                                                           c["adam_beta2"]),
                                                    weight_decay=c["weight_decay"], amsgrad=False)
                    critic_optimizer_S.load_state_dict(models['Critic_optimizer_S'])
                else:
                    gen_optimizer = optim.Adam(self.G.parameters(), lr=c["g_learning_rate"], betas=(c["adam_beta1"],
                                                                                                    c["adam_beta2"]),
                                               weight_decay=c["weight_decay"], amsgrad=False)
                    critic_optimizer_T = optim.Adam(self.D_T.parameters(),
                                                    lr=c["d_learning_rate"], betas=(c["adam_beta1"], c["adam_beta2"]),
                                                    weight_decay=c["weight_decay"], amsgrad=False)
                    critic_optimizer_S = optim.Adam(self.D_S.parameters(),
                                                    lr=c["d_learning_rate"], betas=(c["adam_beta1"], c["adam_beta2"]),
                                                    weight_decay=c["weight_decay"], amsgrad=False)
            else:
                if c["resume"]["status"]:
                    logger.info(f'Loading and resuming SGD optimizer state dict from {c["resume"]["resume_from"]}')
                    gen_optimizer = optim.SGD(self.G.parameters(), lr=c["g_learning_rate"], momentum=c['sgd_momentum'])
                    gen_optimizer.load_state_dict(models['G_optimizer'])
                    critic_optimizer_T = optim.SGD(self.D_T.parameters(), lr=c["g_learning_rate"],
                                                   momentum=c['sgd_momentum'])
                    critic_optimizer_T.load_state_dict(models['Critic_optimizer_T'])
                    critic_optimizer_S = optim.SGD(self.D_S.parameters(), lr=c["d_learning_rate"],
                                                   momentum=c['sgd_momentum'])
                    critic_optimizer_S.load_state_dict(models['Critic_optimizer_S'])
                else:
                    gen_optimizer = optim.SGD(self.G.parameters(), lr=c["g_learning_rate"], momentum=c['sgd_momentum'])
                    critic_optimizer_T = optim.SGD(self.D_T.parameters(), lr=c["g_learning_rate"],
                                                   momentum=c['sgd_momentum'])
                    critic_optimizer_S = optim.SGD(self.D_S.parameters(), lr=c["d_learning_rate"],
                                                   momentum=c['sgd_momentum'])
            # Log gradients
            if self.log_gradients:
                wandb.watch(self.G, log_freq=200, log='gradients')
                wandb.watch(self.D_S, log_freq=200, log='gradients')
                wandb.watch(self.D_T, log_freq=200, log='gradients')

            path = os.path.join(c["results_dir"], "configs.json")
            with open(path, 'w') as fp:
                json.dump(c, fp, indent=1)

            # Data statistics
            num_samples = D_data.shape[0]
            batches_per_epoch = num_samples // c["samples_per_batch"]
            total_iterations = batches_per_epoch * c["num_epochs"]
            # torch.autograd.set_detect_anomaly(True)

            # Run through epochs
            train_start_time = timeit.default_timer()
            remaining_time = "inf"

            for epoch in range(epoch_start, c["num_epochs"] + 1):
                # Adjust the learning rate, if lr scheduling is turned on
                if c["lr_schedule"]["use"] and epoch % c["lr_schedule"]["freq"] == 0:
                    for param_group in gen_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * c['lr_schedule']['factor']
                    logger.info("Changing learning rate")

                for batch_id, (train_data, label_data) in enumerate(D_dl, start=1):
                    self.G.train(True)  # prepare model for training
                    self.D_S.train(True)
                    self.D_T.train(True)

                    # TRAIN CRITIC
                    d_start_time = timeit.default_timer()
                    for criticTrainIdx in range(self.num_D_steps):
                        critic_optimizer_T.zero_grad()
                        critic_optimizer_S.zero_grad()
                        #   Get the losses for backprop.
                        D_loss_T, grad_penalty_T = wasserstein_critic(self.D_T, self.G, c, device, train_data, label_data)
                        D_loss_S, grad_penalty_S = wasserstein_critic(self.D_S, self.G, c, device, train_data, label_data)
                        # Backpropagation.
                        grad_penalty = grad_penalty_T + grad_penalty_S
                        D_loss_no_grad = D_loss_S + D_loss_T - grad_penalty
                        D_loss = D_loss_S + D_loss_T
                        # torch.autograd.set_detect_anomaly(True)
                        D_loss.backward()
                        # update model weights
                        critic_optimizer_T.step()
                        critic_optimizer_S.step()
                    d_time = (timeit.default_timer() - d_start_time) / c["d_steps"]  # Operation timing.

                    g_start_time = timeit.default_timer()
                    for genTrainIdx in range(self.num_G_steps):
                        gen_optimizer.zero_grad()
                        G_loss_T = wasserstein_gen(self.D_T, self.G, c, device, label_data)  # TEMPORAL DISC
                        G_loss_S = wasserstein_gen(self.D_S, self.G, c, device, label_data)  # SPATIAL DISC LOSS
                        G_loss = G_loss_S + G_loss_T    # add a term for scaling T importance
                        if c['mode_seeking']:
                            G_loss += mode_seeking(self.G, c, device, label_data)
                        # backprop
                        G_loss.backward()
                        # update model weights
                        gen_optimizer.step()
                    g_time = (timeit.default_timer() - g_start_time) / c["g_steps"]

                    # Write data to wandb.
                    if total_batch_tally % c["tensorboard_interval"] == 0:
                        wandb.log({"Discriminator Loss: D(x) - D(G(z))": -1 * D_loss_no_grad.data.item(),
                                   "Discriminator Temporal Loss: D(x) - D(G(z))": -1 * D_loss_T.data.item(),
                                   "Discriminator Spatial Loss: D(x) - D(G(z))": -1 * D_loss_S.data.item(),
                                   "Generator Score: D(G(z))": -1 * G_loss.data.item(),
                                   "Computation Times": {"Discriminator Step": d_time,
                                                         "Generator Step": g_time},
                                   "Gradient Penalty": grad_penalty.data.item(),
                                   "batch": total_batch_tally,
                                   })
                        logger.info(
                            "Discriminator loss: {}, Generator loss: {}.".format(-1 * D_loss_no_grad.data.item(),
                                                                                 -1 * G_loss.data.item()))
                    if total_batch_tally % c["save_interval"] == 0:
                        if not c["parallel"]:
                            # wandb.log_artifact(self.G)
                            # wandb.log_artifact(self.D_S)
                            model_dict = {'epoch_num': epoch,
                                          'batch_num': total_batch_tally,
                                          'Generator': self.G.state_dict(),
                                          'Critic': [self.D_S.state_dict(), self.D_T.state_dict()],
                                          'G_optimizer': gen_optimizer.state_dict(),
                                          'Critic_optimizer_S': critic_optimizer_S.state_dict(),
                                          'Critic_optimizer_T': critic_optimizer_T.state_dict()
                                          }
                            torch.save(model_dict, os.path.join(c["results_dir"], "models",
                                                                "model_{}".format(total_batch_tally)))
                        else:
                            model_dict = {'epoch_num': epoch,
                                          'batch_num': total_batch_tally,
                                          'Generator': self.G.module.state_dict(),
                                          'Critic': [self.D_S.module.state_dict(), self.D_T.module.state_dict()],
                                          'G_optimizer': gen_optimizer.state_dict(),
                                          'Critic_optimizer_S': critic_optimizer_S.state_dict(),
                                          'Critic_optimizer_T': critic_optimizer_T.state_dict()
                                          }
                            torch.save(model_dict, os.path.join(c["results_dir"], "models",
                                                                "model_{}".format(total_batch_tally)))

                    if total_batch_tally % c["print_interval"] == 0:
                        logger.info("Completed: Epoch {}/{} \t Batch {}/{} \t Total Batches {}/{} \t Time left {}".format(
                            epoch, c["num_epochs"], batch_id, batches_per_epoch, total_batch_tally, total_iterations,
                            remaining_time))
                    total_batch_tally = total_batch_tally + 1

                total_time = timeit.default_timer() - train_start_time
                time_per_epoch = total_time / (epoch - epoch_start+1)     # fixed this bug
                remaining_time = str(datetime.timedelta(seconds=(c["num_epochs"] - epoch) * time_per_epoch))

        # When training is done, wrap up the wandb
        wandb.finish()