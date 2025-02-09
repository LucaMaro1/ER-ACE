import numpy as np
import torch

class ClassIncrementalSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, n_tasks):
        targets = np.array(dataset.tensors[1])
        self.n_tasks = n_tasks
        self.classes = np.unique(targets)
        self.n_samples = targets.shape[0]
        self.task = None
        self.target_idx = {}
        self.task_samples = None
        self.task_classes = []
        self.task_samples = []

        for label in self.classes:
            self.target_idx[label] = np.squeeze(np.argwhere(targets == label)) # Raggruppo gli indici degli x per cui c(x) = label
            np.random.shuffle(self.target_idx[label])
        
        classes_per_task = self.classes.shape[0] // self.n_tasks
        for t in range(self.n_tasks):
            self.task_classes.append(self.classes[classes_per_task * t : classes_per_task * (t + 1)])

        for t in range(self.n_tasks):
            task_samples = []
            for task in self.task_classes[t]:
                t_indices = self.target_idx[task]
                task_samples.append(t_indices) # Salvo gli indici da restituire
            
            task_samples = np.concatenate(task_samples)
            np.random.shuffle(task_samples)
            self.task_samples.append(task_samples)



    def set_task(self, task):
        self.task = task


    def __iter__(self):
        task_samples = self.task_samples[self.task]
        for item in task_samples:
            yield item


    def __len__(self):
        return self.n_samples // self.n_tasks
