import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from models._resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models._mixstyle import MixStyle


batch_size = 32
learning_rate = 0.001
num_epochs = 10
num_classes = 7

# data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# data (for test run)
full_dataset = datasets.ImageFolder(root="datasets/PACS", transform=transform)

subset_size = int(0.3 * len(full_dataset))
subset_dataset, _ = random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])

train_size = int(0.7 * len(subset_dataset))
test_size = len(subset_dataset) - train_size

train_dataset, test_dataset = random_split(subset_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# initializing model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(num_classes=num_classes, use_mixstyle=True, mixstyle_layers=['layer1', 'layer2'], mixstyle_p=0.5, mixstyle_alpha=0.3).to(device)

# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

correct = 0
total = 0

# training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "resnet_mixstyle_testrun.pth")