import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as torch_transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from EfficientNet import BrainTumorDataset
from EfficientNet import EfficientNet
import argparse
from tqdm import tqdm
from time import time
import os
from EfficientNet import Utils


parser = argparse.ArgumentParser(prog='train',
                                 description="Trains the EfficientNet \
                                     model with given parameters")
    
parser.add_argument("-d", "--device", type=str, default='cpu',
                    help="Specifies the device type")
parser.add_argument("-mv", "--model-version", type=str, default="b0",
                    help="Specifies the model's version",
                    choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'])
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", required=True)
parser.add_argument("-lr", "--learning-rate", type=float, help="Learning rate", required=True)
parser.add_argument("-op", "--optimizer", type=str, help="Optimizer for the model",
                    choices=['adam', 'adamw', 'sgd', 'rmsprop'], required=True)
parser.add_argument("-tb", "--train-batch-size", type=int, help="Number of batches \
                    for each epoch.", required=True)
parser.add_argument("-vb", "--validation-batch-size", type=int, help="Number of \
                    batches for each epoch", required=True)
parser.add_argument("-dp", "--dataset-path", type=str, help="Dataset path \
                    containing the whole data.", required=True)
parser.add_argument("-ci", "--checkpoint-save-interval", type=int, default=5,
                    help="Checkpoint saving interval. State dictionaries \
                        of specified objects will be saved after every n epochs")
parser.add_argument("-r", "--resolution", type=int,
                    help="Resolution of the image. (In case you may need to specify)")
    
args = parser.parse_args()

efficient_net_options = {
    'b0': (1.0, 1.0, 224, .2),
    'b1': (1.0, 1.1, 240, .2),
    'b2': (1.1, 1.2, 260, .3),
    'b3': (1.2, 1.4, 300, .3),
    'b4': (1.4, 1.8, 380, .4),
    'b5': (1.6, 2.2, 456, .4),
    'b6': (1.8, 2.6, 528, .5),
    'b7': (2.0, 3.1, 600, .5)
}

if args.device == "cuda":
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    DEVICE = torch.device("cpu")

model_version = args.model_version

width_mult, depth_mult, res, dropout_rate = efficient_net_options[model_version]

res = res if args.resolution == None else args.resolution

model = EfficientNet(w=width_mult, d=depth_mult, dropout=0.0)

NUM_OF_EPOCHS = args.epochs
LR = float(args.learning_rate)
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
elif args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
elif args.optimizer == "rmsprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)

tr_batch_size = args.train_batch_size
val_batch_size = args.validation_batch_size
dataset_path = args.dataset_path
checkpoint_save_interval = args.checkpoint_save_interval

def train(rank, world_size):

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    utils = Utils()
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=True)
    transforms = torch_transforms.Compose([
        torch_transforms.Resize(res),
        torch_transforms.CenterCrop(res),
        torch_transforms.PILToTensor(),
    ])

    dataset = BrainTumorDataset(dataset_path,
                                transform=transforms, split="train")

    val_size = int(len(dataset) * 0.3)
    train_size = len(dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_steps = len(train_subset)
    validation_steps = len(val_subset)
    train_loader = DataLoader(train_subset, batch_size=tr_batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False, pin_memory=True)

    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}

    LR_SCHEDULER = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=.1, verbose=True)

    start_time = time()

    if not os.path.isdir("artifacts"):
        os.mkdir("artifacts")

    cp = 1

    for epoch in tqdm(range(NUM_OF_EPOCHS)):

        ddp_model.train()

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        train_num_of_correct_preds = 0
        val_num_of_correct_preds = 0

        for image_batch, labels in train_loader:
            with torch.cuda.amp.autocast_mode.autocast():
                image_batch, labels = image_batch.float().to(rank), labels.to(rank)

                output = ddp_model(image_batch)
                loss = loss_fn(output, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer=optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_train_loss += loss.item()
            train_num_of_correct_preds += (output.argmax(dim=1) == labels).type(torch.float).sum().item()

        LR_SCHEDULER.step()
        
        with torch.no_grad():

            ddp_model.eval()

            for image_batch, labels in val_loader:

                with torch.cuda.amp.autocast_mode.autocast():
                    image_batch, labels = image_batch.float().to(rank), labels.to(rank)

                    val_out = ddp_model(image_batch)
                    epoch_val_loss += loss_fn(val_out, labels).item()
                val_num_of_correct_preds += (val_out.argmax(dim=1) == labels).type(torch.float).sum().item()

        average_train_loss = epoch_train_loss / train_steps
        average_val_loss = epoch_val_loss / validation_steps

        train_acc = train_num_of_correct_preds / train_steps
        val_acc = val_num_of_correct_preds / validation_steps

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(average_train_loss)
        history["val_loss"].append(average_val_loss)

        print(f'[INFO] EPOCH: {epoch + 1}/{NUM_OF_EPOCHS}')
        print(f'Train accuracy: {train_acc}, train loss: {average_train_loss}')
        print(f'Validation accuracy: {val_acc}, validation loss: {average_val_loss}')

        if (epoch + 1) % checkpoint_save_interval == 0:
            utils.save_model(model=ddp_model, artifact_path="artifacts/",
                             model_name=f"model_{model_version}_cp_{cp}.pth")
            torch.save(optimizer.state_dict(), "optimizer_cp_{cp}.pth")
            torch.save(LR_SCHEDULER.state_dict(), "lr_scheduler_cp_{cp}.pth")
            torch.save(scaler.state_dict(), "scaler_cp_{cp}.pth")
        cp += 1

    completion_time = time()

    print(f"[Info] total training time {completion_time - start_time}")
    
    utils.plot_history(history=history, artifact_path="artifacts/", 
                       file_name="history.png")  

    utils.save_history(history=history, artifact_path="artifacts/", 
                       file_name="train_history.pth")

    utils.save_model(model=ddp_model, artifact_path="artifacts/", 
                     model_name=f"final_model_{model_version}.pth")
    
    torch.save(optimizer.state_dict(), "optimizer_final_cp_{cp}.pth")
    torch.save(LR_SCHEDULER.state_dict(), "lr_final_scheduler_cp_{cp}.pth")
    torch.save(scaler.state_dict(), "scaler_final_cp_{cp}.pth")


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train,
        args=(world_size,),
        nprocs=world_size,
        join=True)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
