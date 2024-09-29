import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import warnings
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
import argparse
from tqdm import tqdm
from data3 import CustomDataset
import torchvision.transforms as transforms
import wandb
from torch import nn
import torch.nn.functional as F
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,7"

softmax_helper = lambda x: F.softmax(x, 1)

# Command line argument parsing
parser = argparse.ArgumentParser(description='Stenosis Segmentation Training')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--batch-size', type=int, default=6,
                    help='input batch size for training (default: 2)')
parser.add_argument('--encoder-name', type=str, default='efficientnet-b5',
                    help='name of the encoder for the segmentation model (default: efficientnet-b5)')
parser.add_argument('--encoder-weights', type=str, default='imagenet',
                    help='pretrained weights for the encoder (default: imagenet)')
parser.add_argument('--save-path', type=str, default='./',
                    help='path to save the trained model (default: current directory)')
parser.add_argument('--model-type', type=str, default='UnetPlusPlus',
                    help='type of model to use (default: UnetPlusPlus)')
parser.add_argument('--world-size', type=int, default=8,
                    help='number of distributed processes (default: 1)')
parser.add_argument('--lr', type=float, default=8e-4,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=60,
                    help='max epochs')
parser.add_argument('--dist-url', type=str, default='env://',
                    help='url used to set up distributed training (default: env://)')

args = parser.parse_args()

# Setup
warnings.filterwarnings("ignore")

# Function to dynamically select the model based on command-line arguments
def get_model(model_type, encoder_name, encoder_weights):
    model_dict = {
        'DeepLabV3Plus': smp.DeepLabV3Plus,
        'Unet': smp.Unet,
        'UnetPlusPlus': smp.UnetPlusPlus,
        'FPN': smp.FPN,
        'PAN': smp.PAN,
        'Linknet': smp.Linknet,
        'PSPNet': smp.PSPNet,
    }
    
    if model_type in model_dict:
        return model_dict[model_type](
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=2
        )
    else:
        raise ValueError(f"Model type '{model_type}' not recognized. Available types: {list(model_dict.keys())}")

@torch.no_grad()
def test(val_loader, model, device):
    total_samples, total_f1_score, total_dice_score = 0, 0, 0
    for batch_idx, (inputs, masks) in enumerate(tqdm(val_loader, desc='Testing')):
        inputs, masks = inputs.to(device), masks.to(device).long().squeeze(1)
        pred_masks = model(inputs)

        pred_masks = pred_masks[:, 1, :, :] > pred_masks[:, 0, :, :]
        tp = (masks * pred_masks).sum().float()
        fp = ((1 - masks) * pred_masks).sum().float()
        fn = (masks * ~pred_masks).sum().float()

        f1 = tp / (tp + 0.5 * (fp + fn))
        dice = 2 * tp / (2 * tp + fp + fn)
        
        total_samples += inputs.size(0)
        total_f1_score += f1 * inputs.size(0)
        total_dice_score += dice * inputs.size(0)
        
    val_f1_score = total_f1_score / total_samples
    val_dice_score = total_dice_score / total_samples
    print(f"val_f1_score: {val_f1_score}, val_dice_score: {val_dice_score}")
    return val_f1_score


import torch.nn.functional as F

def dice_loss(pred, target, smooth=1):
    pred = F.softmax(pred, dim=1)
    pred = pred[:, 1, :, :]  # Assuming foreground class is at index 1
    target = target.float()

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


def main():
    # model_dir = '/model'
    # output_dir = '/output'

    new_port = 29501
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size)
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        os.environ["WANDB_API_KEY"] = ''
        os.environ["WANDB_MODE"] = "online"
        # Initialize wandb
        wandb.init(project="KPIs",
                   name=args.model_type + '-' + args.encoder_name,
                   config={
                       "lr": args.lr,
                       "epochs": args.epochs,
                       "batch_size": args.batch_size,
                       "model_type": args.model_type,
                       "encoder": args.encoder_name,
                       "gpus": args.world_size
                   }
                   )

    # mean = [0.5987, 0.5245, 0.7011]
    # std = [0.1722, 0.1750, 0.1036]
    mean = [0.6596, 0.5802, 0.7485]
    std = [0.1593, 0.1699, 0.0980]

    # Data loading
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
        # Add more transformations if needed
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # Add more transformations if needed
    ])

    # train_set = CustomDataset( root_dirs = ['data/Task1/val','data/Task1/train'],
    train_set = CustomDataset( root_dirs = 'data/Task1/train',
                                mode="trainval", 
                                image_transform=image_transform, 
                                mask_transform=mask_transform)
    train_sampler = DistributedSampler(train_set)
    print(f'length of train_sampler: {len(train_sampler)*args.world_size}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, sampler=train_sampler, drop_last=True)

    val_set = CustomDataset(root_dirs='data/Task1/val', 
                            mode = 'val',
                            image_transform=image_transform, 
                            mask_transform=mask_transform)
    val_sampler = DistributedSampler(val_set)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, sampler=val_sampler, drop_last=False)
    
    # Model
    model = get_model(args.model_type, args.encoder_name, args.encoder_weights).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Loss functions
    criterion_ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0], device=device))
    criterion_dc = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, smooth=1e-5, do_bg=False)

    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    print(len(train_loader))
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos', final_div_factor=1e5)

    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # # Warm-up percentage (adjust as needed)
    # warmup_pct = 0.1
    #
    # # Warm-up scheduler lambda function
    # def warmup_scheduler(step):
    #     if step / total_steps < warmup_pct:
    #         return step / (total_steps * warmup_pct)
    #     else:
    #         return max((step - total_steps * warmup_pct) / (total_steps * (1 - warmup_pct)), 0)
    #
    # # LambdaLR scheduler for warm-up
    # scheduler = LambdaLR(optimizer, lr_lambda=warmup_scheduler)

    # Training loop
    best_performance = 0.0

    for epoch in range(args.epochs):
        if local_rank == 0:
            print(f"EPOCH: {epoch + 1}/{args.epochs}")
        model.train()
        total_loss = 0
        total_samples = 0

        for batch_idx, (inputs, masks) in enumerate(train_loader):
            inputs, masks = inputs.to(device), masks.to(device).long().squeeze(1)  # Convert masks to LongTensor
            optimizer.zero_grad()
            pred_masks = model(inputs)

            loss_ce = criterion_ce(pred_masks, masks)
            #loss_dice = dice_loss(pred_masks, masks)

            loss_dice = criterion_dc(pred_masks, masks)

            # loss_mask = torch.ones_like(masks, dtype=torch.float32)
            # # 假设像素值为 0 表示背景（黑色），设置背景像素权重为较小值，例如 0.1
            # loss_mask[masks == 0] = 0 #0.1
            # # 假设像素值为 1 表示肾小球（白色），设置肾小球像素权重为较大值，例如 10.0
            # loss_mask[masks == 255] = 1  #10.0
            # loss_dice = criterion_dc(pred_masks, masks, loss_mask)

            # loss = 0.5 * loss_ce + 0.5 * loss_dice

            # if epoch <= 25:
            #     loss = loss_ce
            # else:
            #     loss = loss_dice
            
            ce_weight = 1 * (1-epoch/args.epochs)
            dice_weight = 1 * (epoch/args.epochs)
            # ce_weight = 1
            # dice_weight = 1
            loss = ce_weight * loss_ce + dice_weight * loss_dice

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            if batch_idx % 10 == 0 and local_rank == 0:
                print(f'ce_weight, dice_weight:{ce_weight}, {dice_weight}')
                current_lr = optimizer.param_groups[0]['lr']
                print(f"batch_idx: {batch_idx}, loss_ce: {loss_ce.item()}, loss_dice: {loss_dice.item()}, Current LR: {current_lr:.8f}")


        avg_loss = total_loss / total_samples
        #scheduler.step(avg_loss)
        
        if local_rank == 0:
            wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        
        model.eval()
        if local_rank == 0:
            print("validation")
        val_f1_score = test(val_loader, model, device)
        
        # Gather the results from all processes
        val_f1_score_tensor = torch.tensor([val_f1_score], device=device)
        
        dist.all_reduce(val_f1_score_tensor, op=dist.ReduceOp.SUM)
        
        val_f1_score = val_f1_score_tensor.item() / args.world_size

        if local_rank == 0:
            print(f'******* epoch:{epoch}, val_f1_score:{val_f1_score}!!!')
            wandb.log({"epoch": epoch + 1, "val_f1_score": val_f1_score})
        
        if val_f1_score > best_performance and local_rank == 0:
            # best_performance = val_f1_score
            # if not os.path.exists(args.save_path):
            #     os.makedirs(args.save_path)
            # model_save_path = os.path.join(args.save_path, f"best_model_epoch_{epoch + 1}.pth")
            # torch.save(model.state_dict(), model_save_path)
            # print(f"Model saved to {model_save_path}")
            best_performance = val_f1_score
            model_save_path = args.save_path
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
    
    # Final summary
    if local_rank == 0:
        print(f"finish training, best performance on val set: {best_performance}")
        wandb.run.finish()

if __name__ == '__main__':
    main()
