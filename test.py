import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models._resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(num_classes=7).to(device)
model.load_state_dict(torch.load("resnet_mixstyle_testrun.pth"))
model.eval()

# data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root="datasets/PACS/sketch", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
