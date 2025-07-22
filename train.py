import warnings

import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from config import Config
from data import get_training_data, get_validation_data
from models import *
from utils import *

warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)

def train():
    # Accelerate
    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()
    device = accelerator.device
    config = {
        "dataset": opt.TRAINING.TRAIN_FILE
    }
    accelerator.init_trackers("lensless", config=config)

    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)

    # Data Loader
    train_file = opt.TRAINING.TRAIN_FILE
    val_file = opt.TRAINING.VAL_FILE

    train_dataset = get_training_data(train_file, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=32, drop_last=False, pin_memory=True)
    val_dataset = get_validation_data(val_file, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': opt.TRAINING.ORI})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=16, drop_last=False, pin_memory=True)

    # Model & Loss
    model = LensNet()
    criterion_psnr = torch.nn.MSELoss()
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    # Optimizer & Scheduler
    optimizer_b = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer_b, scheduler_b = accelerator.prepare(optimizer_b, scheduler_b)

    start_epoch = 1
    best_epoch = 1
    best_psnr = 0
    size = len(testloader)
    # training
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()

        for _, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            # get the inputs; data is a list of [target, input, filename]
            inp = data[0]
            tar = data[1]

            # forward
            optimizer_b.zero_grad()
            res = model(inp)

            loss_psnr = criterion_psnr(res, tar)
            loss_ssim = 1 - structural_similarity_index_measure(res, tar, data_range=1)
            loss_lpips = criterion_lpips(res, tar)

            train_loss = loss_psnr + 0.2 * loss_ssim + 0.2 * loss_lpips

            # backward
            accelerator.backward(train_loss)
            optimizer_b.step()

        scheduler_b.step()

        # testing
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr = 0
            ssim = 0
            for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                # get the inputs; data is a list of [targets, inputs, filename]
                inp = data[0]
                tar = data[1]

                with torch.no_grad():
                    res = model(inp)

                res, tar = accelerator.gather((res, tar))

                psnr += peak_signal_noise_ratio(res, tar, data_range=1)
                ssim += structural_similarity_index_measure(res, tar, data_range=1)

            psnr /= size
            ssim /= size

            if psnr > best_psnr:
                # save model
                best_epoch = epoch
                best_psnr = psnr
                save_checkpoint({
                    'state_dict': model.state_dict(),
                }, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR)

            if accelerator.is_local_main_process:
                accelerator.log({
                    "PSNR": psnr,
                    "SSIM": ssim,
                }, step=epoch)

                print(
                    "epoch: {}, PSNR: {}, SSIM: {}, best PSNR: {}, best epoch: {}"
                    .format(epoch, psnr, ssim, best_psnr, best_epoch))

    accelerator.end_training()


if __name__ == '__main__':
    train()
