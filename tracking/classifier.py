import torch
import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torch import nn, optim
from PIL import Image

class CocoDetectionBBox(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform)
        
        self.id_to_supercategory = {cat['id']: cat['supercategory'] for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.supercategories = list(set(self.id_to_supercategory.values()))
        self.supercategory_to_id = {supercategory: i for i, supercategory in enumerate(self.supercategories)}

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        
        img = transforms.ToPILImage()(img)
    
        supercategory_ids = [self.supercategory_to_id[self.id_to_supercategory[ann['category_id']]] for ann in target]
    
        for ann in target:
            bbox = ann['bbox']
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
    
        if self.transform is not None:
            img = self.transform(img)
    
        return img, supercategory_ids



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = CocoDetectionBBox(root='/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/coco/images/train2017', annFile='/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/coco/annotations/instances_train2017.json', transform=transform)
val_dataset = CocoDetectionBBox(root='/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/coco/annotations/val2017', annFile='/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/coco/annotations/instances_val2017.json', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


netc = torchvision.models.resnet50(pretrained=True)

num_classes = len(train_dataset.supercategories)
netc.fc = nn.Linear(netc.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(netc.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netc.to(device)

for epoch in range(num_epochs):
    print("epoch:",epoch)
    netc.train()
    for images, labels in train_loader:
        #print("size of images",images.size())
        #print("size of labels",labels.size())
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = netc(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("loss:",loss)

    netc.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = netc(images)
            _, predictions = torch.max(outputs, 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')

torch.save(netc.state_dict(), 'coco_classifier.pth')
