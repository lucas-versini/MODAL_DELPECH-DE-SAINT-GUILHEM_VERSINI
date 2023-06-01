# MODAL_SAINT-GUILHEM_VERSINI

In order to run the code, several usual packages have to be installed (such as pytorch, transformers, etc.), as well as the clip package (for instance using ```! pip install git+https://github.com/openai/CLIP.git```).

Three folders are necessary:
- One containing 48 folders corresponding the the 48 classes (for instance "train")
- One containing a folder, which contains all the unlabelled images (for instance, "unlabelled/unlabelled/...")
- One containing all test images (for instance "test")

The paths to these folders have to be filled in the file paths.py.

For each of the Python files, the variables that may need to be changed (paths to save files, hyperparameters, ...) are at the end of the file.

Here is a description of the files:

- paths.py: contains the paths to the different images
This file has to be filled at the very beginning

- data.py: contains the definition of a class to handle various datasets

- models.py: contains the definitions of various classes, with each class representing a deep learning model.

- train_ViT.py: trains a ViT model
- submission_ViT.py: creates a submission for a ViT model

- train_convnext.py: trains a ConvNext model
- submission_convnext.py: creates a submission for a ConvNext model

- CLIP_zero_shot.py: creates a submission for a zero shot CLIP model

- train_CLIP_convnext.py: trains a CLIP model based on ConvNext datasets
Requires the ConvNext datasets (generated by train_convnext.py for instance)
- submission_CLIP.py: creates a submission for a CLIP model

- combine_models.py: creates a submission for a zero shot CLIP model and a trained ConvNext model
Requires a trained ConvNext model (generated by train_convnext.py for instance)
