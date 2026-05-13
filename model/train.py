import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader
import sys 
sys.path.append('../preprocessing')
from combined_dataset import get_combined_train, get_combined_test
from model import FERModel 

# Settings
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# Load combined data
print("Loading datasets...")
train_dataset = get_combined_train()
test_dataset = get_combined_test()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model 
device = torch.device('cpu')
model = FERModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    scheduler.step()
    train_acc = 100. * correct / total 
    print(f'Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} Accuracy: {train_acc:.2f}%')

# Save model 
torch.save(model.state_dict(), '../results/fer_model.pth')
print("Model saved to results/fer_model.pth")