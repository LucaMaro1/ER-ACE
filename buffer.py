import torch
import math
import random
import torch.nn as nn
from collections.abc import Iterable
from collections import OrderedDict

class Buffer(nn.Module):
    def __init__(self, device, capacity):
        super().__init__()

        self.names = []
        self.capacity = capacity
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = device

    def add_buffer(self, name, dtype, size):
        tmp = torch.zeros(size=(self.capacity,) + size, dtype=dtype).to(self.device)
        self.register_buffer(name, tmp)
        self.names.append(name)


    def reservoir_sampling_update(self, batch):
        n_elem = batch['x'].size(0) # Elementi nel batch
        place_left = max(0, self.capacity - self.current_index)

        idx_random = torch.FloatTensor(n_elem).to(self.device)
        upper_bound = 0

        # Check sul massimo numero di valori da inserire nelle posizioni libere del Buffer
        if place_left > 0:
            upper_bound = min(place_left, n_elem)
            idx_random[:upper_bound] = torch.arange(upper_bound) + self.current_index

        # Genero gli indici rimanenti
        for i in range(n_elem - upper_bound):
            idx_random[upper_bound + i] = torch.randint(0, self.n_seen_so_far +  + i + 1, (1,), device=idx_random.device)
        idx_random = idx_random.long()

        # Seleziono gli indici validi e dove sostituirli nel buffer
        valid_indices = (idx_random < self.capacity).long()
        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer = idx_random[idx_new_data]

        self.n_seen_so_far += n_elem
        self.current_index = min(self.n_seen_so_far, self.capacity)

        if idx_buffer.numel() == 0: return # Nessun indice valido

        # Sostituisco i dati nel Buffer
        for name, data in batch.items():
            buffer = getattr(self, f'b{name}')

            if isinstance(data, Iterable):
                buffer[idx_buffer] = data[idx_new_data]
            else:
                buffer[idx_buffer] = data


    def sample(self, re_batch_size):
        buffers = OrderedDict()
        for buffer_name in self.names:
            buffers[buffer_name[1:]] = getattr(self, buffer_name)[:self.current_index] # Recupero tutti i dati fino all'indice corrente

        n_selected = buffers['x'].size(0)
        if n_selected <= re_batch_size:
            return buffers # Posso restituire tutti gli elementi del Buffer

        labels = torch.unique(buffers['y'])
        samples_per_class = math.ceil(re_batch_size / len(labels))
        class_indices = {cls.item(): (buffers['y'] == cls).nonzero(as_tuple=True)[0] for cls in labels}

        # Sampling
        selected_indices = []
        for _, indices in class_indices.items():
            if len(indices) >= samples_per_class:
                selected_indices.append(indices[torch.randperm(len(indices))[:samples_per_class]])
            else:
                selected_indices.append(indices)

        # Elimina gli indici in ecceso
        if (len(selected_indices) * len(selected_indices[0])) > re_batch_size:
            surplus = (len(selected_indices) * len(selected_indices[0])) - re_batch_size
            drop_in_classes = random.sample(range(len(labels)), surplus)
            for i in drop_in_classes:
                selected_indices[i] = selected_indices[i][:-1]

        selected_indices = torch.cat(selected_indices)

        return OrderedDict({k: v[selected_indices] for k, v in buffers.items()}) # k = ['x', 'y']
                                                                                 # v --> tensori contenenti le immagini e le label


    def empty(self):
        for name in self.names:
            bfr = getattr(self, f'{name}')
            tmp = torch.zeros(size=bfr.size(), dtype=bfr.dtype).to(self.device)
            self.register_buffer(name, tmp)
        self.current_index = 0
        self.n_seen_so_far = 0


    def n_bits(self):
        total = 0

        for name in self.names:
            buffer = getattr(self, name)

            if buffer.dtype == torch.float32:
                bits_per_item = 8 if name == 'bx' else 32
            elif buffer.dtype == torch.int64:
                bits_per_item = buffer.max().float().log2().clamp_(min=1).int().item()

            total += bits_per_item * buffer.numel()

        return total


    def __len__(self):
        return self.current_index