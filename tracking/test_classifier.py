import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image

# Load the pre-trained ResNet model
model = resnet50(pretrained=True)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image path
image_path = '/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/nfs/airboard_1/30/airboard_1/00001.jpg'
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)

# Make a prediction
with torch.no_grad():
    output = model(input_batch)

# Load ImageNet class labels from the file
with open("imagenet_classes.txt", "r") as f:
    labels = eval(f.read())

# Get the predicted class index
predicted_class_index = torch.argmax(output[0]).item()

# Get the predicted class labels
predicted_class_labels = labels.get(str(predicted_class_index), ["Unknown"])

# Convert predicted_class_labels to a tensor
class_labels_tensor = torch.tensor(predicted_class_labels)
