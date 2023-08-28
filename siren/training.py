"""Implements a generic training loop.
"""

import os
import shutil
import time

import numpy as np
import torch
import utils
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm


def train(
    model,
    train_dataloader,
    epochs,
    lr,
    steps_til_summary,
    epochs_til_checkpoint,
    model_dir,
    loss_fn,
    summary_fn,
    wandb,
    val_dataloader=None,
    double_precision=False,
    clip_grad=False,
    use_lbfgs=False,
    loss_schedules=None,
    filename=None,
    cfg=None,
):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    if cfg.scheduler.type == "step":
        scheduler = StepLR(
            optim,
            step_size=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma,
            verbose=True,
        )
    elif cfg.scheduler.type == "adaptive":
        scheduler = ReduceLROnPlateau(
            optim,
            patience=cfg.scheduler.patience_adaptive,
            factor=cfg.scheduler.factor,
            verbose=True,
            threshold=cfg.scheduler.threshold,
            min_lr=cfg.scheduler.min_lr,
        )

    # copy settings from Raissi et al. (2019) and here
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(
            lr=lr,
            params=model.parameters(),
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

    # if os.path.exists(model_dir):
    #     val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
    #     if val == 'y':
    #         shutil.rmtree(model_dir)

    os.makedirs(model_dir, exist_ok=True)
    checkpoints_dir = model_dir

    # writer = SummaryWriter(summaries_dir)
    if cfg.augment_on_the_fly:
        os.makedirs(checkpoints_dir + "_aug", exist_ok=True)
    total_steps = 0
    best_loss = float("inf")
    patience = cfg.scheduler.patience
    num_bad_epochs = 0

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            # if not epoch % epochs_til_checkpoint and epoch:
            #     torch.save(model.state_dict(),
            #                os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
            #                np.array(train_losses))
            total_loss, total_items = 0, 0

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {
                        key: value.double() for key, value in model_input.items()
                    }
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:

                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.0
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean()
                        train_loss.backward()
                        return train_loss

                    optim.step(closure)

                model_output = model(model_input)
                losses = loss_fn(model_output, gt, model)

                train_loss = 0.0
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        # writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        # wandb.log({loss_name + "_weight": loss_schedules[loss_name](total_steps)})
                        single_loss *= loss_schedules[loss_name](total_steps)

                    # writer.add_scalar(loss_name, single_loss, total_steps)
                    # wandb.log({loss_name: single_loss})
                    train_loss += single_loss
                    # wandb.log({loss_name: single_loss})
                train_losses.append(train_loss.item())
                total_loss += train_loss.item() * len(model_output)
                total_items += len(model_output)
                # writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    # torch.save(model.state_dict(),
                    #            os.path.join(checkpoints_dir, f'{filename}_model_current.pth'))
                    pass

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1.0
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=clip_grad
                            )

                    optim.step()

                pbar.update(1)
                # wandb.log({"loss": train_loss})
                pbar.set_description(
                    "Epoch %d, Total loss %0.6f, iteration time %0.6f"
                    % (epoch, train_loss, time.time() - start_time)
                )
                if not total_steps % steps_til_summary:
                    # pbar.set_description("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for model_input, gt in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            # writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                            wandb.log({"val_loss": np.mean(val_losses)})
                        model.train()
                total_steps += 1
            epoch_loss = total_loss / total_items
            if cfg.scheduler.type == "adaptive":
                prev_bad_epochs = scheduler.num_bad_epochs
                prev_best = scheduler.best
                prev_lr = next(iter(optim.param_groups))["lr"]
                scheduler.step(epoch_loss)
                curr_bad_epochs = scheduler.num_bad_epochs
                new_lr = next(iter(optim.param_groups))["lr"]
                new_best = scheduler.best
            else:
                scheduler.step()
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
            optim.param_groups[0]["lr"] = max(
                optim.param_groups[0]["lr"], cfg.scheduler.min_lr
            )
            if cfg.strategy == "continue":
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        checkpoints_dir, f"{filename}_model_final_{epoch}.pth"
                    ),
                )

            wandb.log(
                {
                    "epoch_loss": epoch_loss,
                    "lr": optim.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )

            if num_bad_epochs == patience:
                break
        if not cfg.mlp_config.move:
            summary_fn(
                "audio_samples",
                model,
                model_input,
                gt,
                model_output,
                wandb,
                total_steps,
            )
        wandb.log({"total_train_loss": train_loss.item()})
        if cfg.strategy != "continue":
            torch.save(
                model.state_dict(),
                os.path.join(checkpoints_dir, f"{filename}_model_final.pth"),
            )
        # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
        #            np.array(train_losses))


class LinearDecaySchedule:
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(
            iter / self.num_steps, 1.0
        )
