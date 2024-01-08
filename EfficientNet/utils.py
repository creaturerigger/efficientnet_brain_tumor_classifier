import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from collections import OrderedDict

class Utils:

    def __init__(self):
        pass

    def plot_history(self, history, artifact_path, file_name):
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Loss and Accuracy Chart')
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss/Acc')
        plt.ylim(top=1.1)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend(loc='center right')
        plt.savefig(os.path.join(artifact_path, file_name))


    def save_model(self, model, artifact_path, model_name):
        torch.save(model.module.state_dict(), os.path.join(artifact_path, model_name))


    def save_history(self, history: dict, artifact_path, file_name):
        or_dict = OrderedDict(history)
        torch.save(or_dict, os.path.join(artifact_path, file_name))