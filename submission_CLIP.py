import os

import pandas as pd
from PIL import Image

from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

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

def create_submission(train_path, test_path, test_transform, batch_size, num_classes = 10):
    test_loader = DataLoader(
        TestDataset(test_path, test_transform),
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0,
    )

    # Load model and checkpoint
    model = CLIPFinetune(num_classes, class_tokens, device, False).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    class_names = sorted(os.listdir(train_path))

    # Create submission.csv
    submission = pd.DataFrame(columns = ["id", "label"])

    for i, batch in enumerate(test_loader):
        images, image_names = batch
        images = images.to(device)
        preds = model(images)
        preds = preds.max(1).values
        preds = preds.detach().cpu().numpy()
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"id": image_names, "pred": preds}),
            ]
        )
    submission.to_csv(f"submission_CLIP.csv", index=False)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

class_names = sorted(os.listdir(train_path))
class_names = ["An image of " + x for x in class_names]
class_tokens = clip.tokenize(class_names).to(device)

num_classes = 48
batch_size = 50

model_path = "model_CLIP.pth"

create_submission(train_path,
                    test_path,
                    transform,
                    batch_size,
                    num_classes)