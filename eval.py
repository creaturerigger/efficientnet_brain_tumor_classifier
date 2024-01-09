import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchvision.transforms as torch_transforms
from EfficientNet import Utils
from EfficientNet import EfficientNet
from EfficientNet import BrainTumorDataset
from tqdm import tqdm
from collections import OrderedDict
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(prog="eval",
                                 description="This program evaluates the model \
                                    on a given dataset.")

parser.add_argument("-dp", "--dataset-path", type=str,
                    help="Path to the test dataset.",
                    required=True)
parser.add_argument("-tbs", "--test-batch-size", type=int,
                    help="Batch size for test dataset.", required=True)
parser.add_argument("-mp", "--model-path", type=str,
                    help="Checkpoint path for the model.",
                    required=True)
parser.add_argument("-mv", "--model-version", type=str, default="b0",
                    help="Specifies the model's version",
                    choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'])
parser.add_argument("-nc", "--number-of-classes", type=int,
                    help="Number of output classes.", required=True)

args = parser.parse_args()

efficient_net_options = {
    'b0': (1.0, 1.0, 224, .2),
    'b1': (1.0, 1.1, 240, .2),
    'b2': (1.1, 1.2, 260, .3),
    'b3': (1.2, 1.4, 300, .3),
    'b4': (1.4, 1.8, 380, .4),
    'b5': (1.6, 2.2, 456, .4),
    'b6': (1.8, 2.6, 528, .5),
    'b7': (2.0, 3.1, 380, .5)
}


def get_confusion_matrix(y_pred, y_true):
    num_classes = len(np.unique(y_true))
    conf_mat = torch.zeros((num_classes, num_classes))
    for i in range(len(y_true)):      
        true_class_idx = y_true[i]
        idx = y_pred[i]
        conf_mat[true_class_idx, idx] += 1
    return conf_mat


def eval(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)
    if not os.path.isdir('artifacts'):
        os.mkdir('artifacts')

    dataset_path = args.dataset_path
    batch_size = args.test_batch_size
    checkpoint = args.model_path
    model_version = args.model_version
    num_classes = args.number_of_classes
    width_coef, depth_coef, res, _ = efficient_net_options[model_version]
    transforms = torch_transforms.Compose([
        torch_transforms.Resize(res),
        torch_transforms.CenterCrop(res),
        torch_transforms.PILToTensor(),
    ])

    test_dataset = BrainTumorDataset(dataset_path, 
                                     transform=transforms,
                                     target_transform=None, 
                                     split='test')
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)
    
    model_dict = torch.load(checkpoint, map_location=torch.device(rank))
    test_model = EfficientNet(w=width_coef, d=depth_coef,
                              num_classes=num_classes)
    test_model.load_state_dict(model_dict)
    
    test_model.to(rank)
    ddp_model = DDP(test_model, device_ids=[rank])
    ddp_model.eval()

    wrong_images = []
    wrong_predictions = []
    correct_labels = []
    predictions = []
    targets = []
    scores = []
    
    with torch.no_grad():
        
        for image_batch, labels in tqdm(test_loader):
            image_batch = image_batch.float().to(rank)
            labels = labels.to(rank)
            output = ddp_model(image_batch)
            _, predicted_labels = torch.max(output, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
            scores.extend(output.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            for i in range(len(output)):
                if predicted_labels[i] != labels[i]:
                    wrong_images.append(image_batch[i].cpu().numpy())
                    wrong_predictions.append(predicted_labels[i])
                    correct_labels.append(labels[i])
    conf_mat = get_confusion_matrix(predictions, test_dataset.labels)
    eval_dict = OrderedDict([('wrong_images', wrong_images),
                               ('wrong_predictions', wrong_predictions),
                               ('correct_labels', correct_labels),
                               ('confusion_matrix', conf_mat)])
    # dist.barrier()
    torch.save(eval_dict, 'artifacts/eval_dict.pth')
    correct_preds = np.equal(test_dataset.labels, predictions)
    test_acc = np.mean(correct_preds) * 100.0
    print("Model's accuracy on test set is %.3f"%test_acc)


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(eval,
        args=(world_size,),
        nprocs=world_size,
        join=True)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "4718"
    main()