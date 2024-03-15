import argparse
import os
from utils import get_args
import ipdb
import torch
torch.cuda.empty_cache()

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.3.1: Implement GAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    """
    # https://neptune.ai/blog/gan-loss-functions
    # loss_fn = torch.nn.Sigmoid()
    # discrim_real = loss_fn(discrim_real)
    # discrim_fake = loss_fn(discrim_fake)
    # # ipdb.set_trace()
    # bcewithlogitloss = torch.nn.BCEWithLogitsLoss()
    # loss = bcewithlogitloss(discrim_fake, torch.ones_like(discrim_fake)) + bcewithlogitloss(discrim_real, torch.zeros_like(discrim_real))
    # loss = torch.log(discrim_real) + torch.log(1 - discrim_fake)
    # loss = torch.sum(loss)/loss.shape[0]

    loss = F.binary_cross_entropy_with_logits(discrim_real, torch.ones_like(discrim_real)) + \
           F.binary_cross_entropy_with_logits(discrim_fake, torch.zeros_like(discrim_fake))

    return loss


def compute_generator_loss(discrim_fake):
    """
    TODO 1.3.1: Implement GAN loss for generator.
    """
    # loss_fn = torch.nn.Sigmoid()
    # discrim_fake = loss_fn(discrim_fake)
    # # loss = torch.log(1 - discrim_fake)
    # # loss = torch.sum(loss)/loss.shape[0]
    # # loss = loss.half()
    # bcewithlogitloss = torch.nn.BCEWithLogitsLoss()
    # loss = bcewithlogitloss(discrim_fake, torch.ones_like(discrim_fake))
    loss = F.binary_cross_entropy_with_logits(discrim_fake, torch.ones_like(discrim_fake))
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256//16,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=100,
        amp_enabled=not args.disable_amp,
    )
