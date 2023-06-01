import os
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from torchvision.datasets import ImageFolder, DatasetFolder

import numpy as np

from models import *
from paths import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)

def create_submission(train_path, test_path, test_transform, batch_size, checkpoint_path, num_classes, lambda_ = 0.1):
    train_dataset = ImageFolder(train_path, transform = test_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0,
    )
    test_loader = DataLoader(
        TestDataset(test_path, test_transform),
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0,
    )

    # Load model and checkpoint
    model1 = ConvNextFinetune(num_classes).to(device)
    checkpoint = torch.load(checkpoint_path)
    model1.load_state_dict(checkpoint)
    model1.eval()

    model2 = CLIPFinetune(num_classes, class_tokens, device, False).to(device)
    model2.eval()

    num_correct = 0
    num_samples = 0
    for i, batch in enumerate(train_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        preds1 = torch.nn.Softmax(dim = 1)(model1(images))
        preds2 = model2(images)

        preds = lambda_ * preds1 + (1 - lambda_) * preds2

        num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
        num_samples += len(images)
    accuracy = num_correct / num_samples
    print(f"lambda = {lambda_}, accuracy = {accuracy}")

    # Create submission.csv
    submission = pd.DataFrame(columns = ["id", "label"])

    for i, batch in enumerate(test_loader):
        images, image_names = batch
        images = images.to(device)

        preds1 = torch.nn.Softmax(dim = 1)(model1(images))
        preds2 = model2(images)

        preds = lambda_ * preds1 + (1 - lambda_) * preds2

        preds = preds.argmax(1)
        preds = [class_names[pred] for pred in preds.cpu().numpy()]
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"id": image_names, "label": preds}),
            ]
        )
    submission.to_csv("combine_models.csv", index=False)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])


num_classes = 48
batch_size = 10

convnext_path = "path/to/convnext_model.pth"


class_names = sorted(os.listdir(train_path))
class_tokens = clip.tokenize(["An image of " + x for x in class_names]).to(device)

create_submission(train_path,
                    test_path,
                    transform,
                    batch_size,
                    convnext_path,
                    num_classes)
