import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from mmfnet import MMFNet
from configs.train_configs import MMFNetConfig
from utils.loss_fn import cross_entropy_loss_fn, focal_loss_fn, ghmc_loss_fn, fringe_loss_fn


def train_fn(model: MMFNet, train_config: MMFNetConfig,
             train_dataset, write_log=True):
    print(f"Start Training on Seed {train_config.seed}")
    model = model.to(train_config.device)

    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size,
                              shuffle=True, num_workers=4)

    if write_log:
        log_dir = train_config.log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    model_optim = torch.optim.Adam(model.parameters(), lr=train_config.model_lr)
    model_optim_scheduler = torch.optim.lr_scheduler.StepLR(model_optim,
                                                            step_size=train_config.decay_steps,
                                                            gamma=train_config.decay_gamma)

    if train_config.class_loss_fn == "ce":
        loss_fn = cross_entropy_loss_fn(reduction="mean")
    elif train_config.class_loss_fn == "focal":
        loss_fn = focal_loss_fn(gamma=2, alpha=0.25, reduction="mean")
    elif train_config.class_loss_fn == "ghmc":
        loss_fn = ghmc_loss_fn(bins=10, momentum=0, reduction="mean")
    elif train_config.class_loss_fn == "fringe":
        loss_fn = fringe_loss_fn(eps=0.25, reduction="mean")
    else:
        raise ValueError(f"Invalid class loss function {train_config.class_loss_fn}")

    device = train_config.device

    epoch_num = train_config.epoch_num

    for i_epoch in range(epoch_num):
        train_loop(model, model_optim, model_optim_scheduler,
                   loader=train_loader, device=device, writer=writer,
                   loss_fn=loss_fn, i_epoch=i_epoch)

    state_dict = model.state_dict()

    # write log
    if write_log:
        torch.save(state_dict, f"{log_dir}/mmfnet.pth")
        writer.close()


def train_loop(model: MMFNet, model_optim, model_optim_scheduler,
               loader, writer: SummaryWriter, loss_fn, device: str,
               current_epoch: int):
    print("Training Loop")
    model.train()

    num_step = current_epoch * len(loader)

    pbar = tqdm(total=len(loader), desc=f"Epoch {current_epoch}")
    for i_step, (images, structs, labels) in enumerate(loader):
        images, structs, labels = images.to(device), structs.to(device), labels.to(device)

        y_logits, hidden = model.forward_with_hidden(images)
        loss = loss_fn(y_logits, labels)

        loss.backward()
        model_optim.step()
        lr = model_optim.param_groups[0]['lr']
        model_optim_scheduler.step()

        # eval
        y_score = torch.sigmoid(y_logits).detach().cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)
        labels = labels.cpu().numpy().astype(np.int8)

        roauc = roc_auc_score(y_true=labels, y_score=y_score)
        f1 = f1_score(y_true=labels, y_pred=y_pred)

        # write log
        if writer is not None:
            log_dict = {"class_loss": loss.item(),
                        "lr": lr,
                        "roauc": roauc,
                        "f1": f1}
            for key, value in log_dict.items():
                writer.add_scalar(f"train/{key}", value, num_step)

        num_step += 1
        pbar.update()
        pbar.write(f"Step {i_step + 1}: roauc {roauc}, F1 {f1}, Loss {loss.item()}")
