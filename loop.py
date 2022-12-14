import torch


def return_device():
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    return device

def train_loop(dataloader, model, loss_fn, optimizer, logger=None):
    device = return_device()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(device))
        loss = loss_fn(pred.to('cpu'), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if logger == None:
        #     graph_trn_loss.append(loss)

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def vaild_loop(dataloader, model, loss_fn, logger=None):
    device = return_device()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device)).to('cpu')
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size


    # if logger == None:
    #     graph_vad_loss.append(test_loss)
    #     graph_vad_acc.append(100*correct)
    
    print(f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def test_loop(dataloader, model, loss_fn, logger=None):
    device = return_device()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device)).to('cpu')
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    
    # if logger == None:
    #     graph_tst_loss.append(test_loss)
    #     graph_tst_acc.append(100*correct)
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
