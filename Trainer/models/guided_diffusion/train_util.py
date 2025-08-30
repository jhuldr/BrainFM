import copy
import functools
import os

import math
import numpy as np
import nibabel as nib
import blobfile as bf
import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

import dist_util, logger
from fp16_util import MixedPrecisionTrainer
from nn import update_ema
from resample import LossAwareSampler, UniformSampler
 

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        classifier,
        diffusion,
        data,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.dataloader = dataloader
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if torch.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

 

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print("resume model")
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        i = 0
        data_iter = iter(self.dataloader)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            try:
                batch, cond, path, slicedict = next(data_iter) # (b=1, 2, 224, 224, 224), (b=1, 1, 224, 224, 224)
                print('batch, cond', batch.shape, cond.shape)

                batch_size_vol = 16
                nr_batches = len(slicedict) / batch_size_vol # only input images within to-inpaint masks

                nr_batches = math.ceil(nr_batches)

                for b in range(0, nr_batches):
                    out_batch = []
                    out_cond = []

                    print('slicedict', slicedict)
                    print('slicedict', len(slicedict), b, nr_batches)
                    if len(slicedict) > b * batch_size_vol + batch_size_vol:
                        print('in', len(slicedict), b * batch_size_vol + batch_size_vol)
                        for s in slicedict[
                            b * batch_size_vol : (b * batch_size_vol + batch_size_vol)
                        ]:
                            print('s', s)
                            out_batch.append(torch.tensor(batch[..., s])) # (b=1, 2, w, h)
                            out_cond.append(torch.tensor(cond[..., s])) # (b=1, 1, w, h)

                        out_batch = torch.stack(out_batch) # (batch_size_vol, b=1, 2, w, h)
                        out_cond = torch.stack(out_cond) # (batch_size_vol, b=1, 1, w, h)

                        print('1 out_batch, out_cond', out_batch.shape, out_cond.shape)
 
                        out_batch = out_batch.squeeze(1)
                        out_cond = out_cond.squeeze(1)

                        out_batch = out_batch.squeeze(4) # (batch_size_vol, 2, w, h)
                        out_cond = out_cond.squeeze(4) # (batch_size_vol, 2, w, h)
                        print('2 out_batch, out_cond', out_batch.shape, out_cond.shape)
 

                        p_s = path[0].split("/")[3]

                        self.run_step(out_batch, out_cond)

                        i += 1

                    else:
                        print('not in', len(slicedict), b * batch_size_vol + batch_size_vol)
                        for s in slicedict[b * batch_size_vol :]:
                            print('s', s)
                            out_batch.append(torch.tensor(batch[..., s]))
                            out_cond.append(torch.tensor(cond[..., s]))

                        out_batch = torch.stack(out_batch)
                        out_cond = torch.stack(out_cond)

                        out_batch = out_batch.squeeze(1)
                        out_cond = out_cond.squeeze(1)
                        out_batch = out_batch.squeeze(4) # (< batch_size_vol, 2, w, h)
                        out_cond = out_cond.squeeze(4) # (< batch_size_vol, 2, w, h)
                        print('NOT out_batch, out_cond', out_batch.shape, out_cond.shape)

                        p_s = path[0].split("/")[3]

                        self.run_step(out_batch, out_cond)

                        i += 1

            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                data_iter = iter(self.dataloader)

                batch, cond, path, slicedict = next(data_iter)

                batch_size_vol = 16
                nr_batches = len(slicedict) / batch_size_vol

                nr_batches = math.ceil(nr_batches)

                for b in range(0, nr_batches):
                    out_batch = []
                    out_cond = []

                    if len(slicedict) > b * batch_size_vol + batch_size_vol:
                        for s in slicedict[
                            b * batch_size_vol : (b * batch_size_vol + batch_size_vol)
                        ]:
                            out_batch.append(torch.tensor(batch[..., s]))
                            out_cond.append(torch.tensor(cond[..., s]))

                        out_batch = torch.stack(out_batch)
                        out_cond = torch.stack(out_cond)

                        out_batch = out_batch.squeeze(1)
                        out_cond = out_cond.squeeze(1)
                        out_batch = out_batch.squeeze(4)
                        out_cond = out_cond.squeeze(4)

                        p_s = path[0].split("/")[3]

                        self.run_step(out_batch, out_cond)

                        i += 1

                    else:
                        for s in slicedict[b * batch_size_vol :]:
                            out_batch.append(torch.tensor(batch[..., s]))
                            out_cond.append(torch.tensor(cond[..., s]))

                        out_batch = torch.stack(out_batch)
                        out_cond = torch.stack(out_cond)

                        out_batch = out_batch.squeeze(1)
                        out_cond = out_cond.squeeze(1)
                        out_batch = out_batch.squeeze(4)
                        out_cond = out_cond.squeeze(4)

                        p_s = path[0].split("/")[3]

                        self.run_step(out_batch, out_cond)

                        i += 1

            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        print("batch pre:", batch.shape) # (16, 2, 224, 224) # void + mask
        print("cond  pre:", cond.shape) # (16, 1, 224, 224) # all (unmasked)
        batch = torch.cat((batch, cond), dim=1) # (16, 3, 224, 224)
        print("batch:", batch.shape) # (16, 3, 224, 224)
        cond = {}
        sample = self.forward_backward(batch, cond) 
        print("out sample:", sample.shape) # (16, 1, 224, 224)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return sample

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()

        micro = batch.to(dist_util.dev())
        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev()) # (16, )

        print('micro, t, weights:', micro.shape, t.shape, weights.shape)

        compute_losses = functools.partial(
            self.diffusion.training_losses_segmentation,
            self.ddp_model, # UNet
            self.classifier, # None
            micro, # (16, 3, 224, 224)
            t,
        )

        #    if last_batch or not self.use_ddp:
        losses1 = compute_losses()

        # else:
        #       with self.ddp_model.no_sync():
        #                    losses1 = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())
        losses = losses1[0]
        sample = losses1[1]

        print('--- losses', losses)

        loss = (losses["loss"] * weights).mean()
        print('--- avg loss', loss)

        log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
        self.mp_trainer.backward(loss)
        return sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = (
                        f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                    )
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    torch.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(
                    get_blob_logdir(),
                    f"optsavedmodel{(self.step+self.resume_step):06d}.pt",
                ),
                "wb",
            ) as f:
                torch.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


 