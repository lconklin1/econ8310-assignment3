import torch
import torch.nn as nn
import torch.nn.functional as F

def eval_model(PATH, model, test_dataloader):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    EPOCH = checkpoint['epoch']

    model.eval()  # Puts model in evaluation mode (no dropout, etc.)

    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed for inference
        for images, labels in test_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return print(f'Test Accuracy: {100 * correct / total:.2f}%')