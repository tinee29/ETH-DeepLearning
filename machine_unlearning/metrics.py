import torch

def accuracy_loss(model, dataloader):
    with torch.no_grad():
        correct = 0
        loss = 0
        for data in dataloader:
            x, y = data
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            pred = model(x) 
            
            predicted = torch.argmax(pred, dim=-1)
            correct += (predicted == y).int().sum()
            
            loss += torch.nn.CrossEntropyLoss()(pred.squeeze(), y.squeeze())            

    return correct/len(dataloader.dataset), loss/len(dataloader.dataset)
