import torch
import copy
import time
import numpy as np

from dataset import transform

def eval_model(model, dataloader, task, device, verbose):
    model.eval()

    accs = np.zeros(shape=(dataloader.sampler.n_tasks,))

    with torch.no_grad():
        for t in range(task + 1):
            n_correct_preds, n_total = 0, 0
            dataloader.sampler.set_task(t)

            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                pred = model(x).max(1)[1] # Seleziono la classe che ha il valore più alto

                n_correct_preds += pred.eq(y).sum().item()
                n_total += x.size(0)

            accs[t] = n_correct_preds / n_total * 100 # Accuracy per la task specifica

        avg_acc = np.mean(accs[:task + 1]) # Accuracy generale
        if verbose: print('\nValidation Results:\t', '\t'.join([str(int(x)) for x in accs]), f'\tAvg Acc: {avg_acc:.2f}\n')

    return accs


def training_gs(original_model, criterion, n_tasks, train_loader, val_loader, buffer, re_batch_size, aug_flag, device, verbose):
    # Training con gridsearch
    # Parametri e valori presi dal paper originale

    lr_gs = [0.1, 0.01, 0.001]
    print(f"Training with grid search on learning rate ({lr_gs})")

    original_model = original_model.to(device)

    best_model = original_model
    best_accuracy = 0
    best_lr = lr_gs[0]
    best_eval_accs = []

    for lr in lr_gs:
        model = copy.deepcopy(original_model)
        classes_seen_so_far = torch.LongTensor(size=(0,)).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        eval_accs = []

        if verbose: print(f"\n\nlearning rate = {lr}:")

        for tsk in range(n_tasks):
            train_loader.sampler.set_task(tsk)
            model.train()

            if verbose: print(f"\nTask {tsk}:\n")

            start = time.time()
            for i, (x,y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                if aug_flag:
                    aug_x = transform(x)
                else:
                    aug_x = x
                inc_data = {'x': x, 'y': y}

                # Calcolo loss per i dati del batch
                batch_classes = y.unique()
                classes_seen_so_far = torch.cat([classes_seen_so_far, batch_classes]).unique()
                logits = model(aug_x)
                mask = torch.zeros_like(logits)
                mask[:, batch_classes] = 1 # Sblocco gli elementi della maschera per le classi contenute nel batch
                mask[:, classes_seen_so_far.max():] = 1 # Sblocco gli elementi della maschera per le classi non viste
                logits = logits.masked_fill(mask == 0, -1e9)
                inc_loss = criterion(logits, y.long())

                re_loss  = 0
                if len(buffer) > 0:
                    # Calcolo loss per i dati campionati dal buffer
                    re_data = buffer.sample(re_batch_size)
                    re_logits = model(re_data['x'])
                    re_loss = criterion(re_logits, re_data['y'].long())

                loss = inc_loss + re_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                buffer.reservoir_sampling_update(inc_data) # Aggiorno gli elementi nel buffer

            if (i+1) == len(train_loader): # Ultima iterazione
                if verbose: print(f"Time {time.time() - start:.2f}")
                acc = eval_model(model, val_loader, tsk, device, verbose)
                model.train()

                eval_accs += [acc]

        buffer.empty()

        final_acc = eval_accs[-1].mean()
        print(f"lr = {lr}, final accuracy = {final_acc:.4f}")
        if final_acc > best_accuracy:
            best_model = model
            best_lr = lr
            best_accuracy = final_acc
            best_eval_accs = eval_accs

    print(f"\nBest learning rate = {best_lr}")
    return best_eval_accs, best_model