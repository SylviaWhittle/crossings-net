"""Scripts for training the model."""

from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.resunet import ResUNet
from src.loss import PermutationInvariantDiceLoss_2_channel

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "resunet_model.pth"
BASE_DIR = Path("/Users/sylvi/topo_data/crossings_net")
PREDICTIONS_DIR = BASE_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)  # create the directory if it doesn't exist

SAMPLE_TYPES = [
    "false",
    "true_slot",
]

NUM_SAMPLES_PER_TYPE = [100, 0]  # number of samples to grab from each sample type
TRAINING_DATA_DIR = BASE_DIR / "training"
ALLOW_TOO_FEW_SAMPLES = True


def sample_type_split(
    sample_type_dirs: list[Path],
    num_samples_per_type: list[int],
    allow_too_few_samples: bool = False,
) -> tuple[list[Path], list[Path]]:
    """
    Grab images from each sample type for the dataset.

    Parameters
    ----------
    sample_type_dirs : list[Path]
        List of directories containing the sample types.
    num_samples_per_type : list[int]
        List of the number of samples to grab from each sample type.
    allow_too_few_samples : bool, optional
        Whether to allow fewer samples than requested if a sample type has too few samples, by default False.

    Returns
    -------
    tuple[list[Path], list[Path]]
        A tuple containing two lists: the first list contains the paths to the sampled image files,
        and the second list contains the paths to the corresponding mask files.
    """
    assert len(sample_type_dirs) == len(
        num_samples_per_type
    ), "sample_type_dirs and num_samples_per_type must have the same length"

    sampled_image_files = []
    sampled_mask_files = []

    print(f"Sampling {num_samples_per_type} files from each sample type directory in {sample_type_dirs}.")

    for sample_type_dir, num_samples in zip(sample_type_dirs, num_samples_per_type):
        # get all the images and masks in the sample type directory
        image_files = sorted(list((sample_type_dir / "images").glob("*.npy")))
        mask_files = sorted(list((sample_type_dir / "labels").glob("*.npy")))

        print(f"| Found {len(image_files)} images and {len(mask_files)} masks in {sample_type_dir}.")
        assert len(image_files) == len(mask_files), f"Number of images and masks must be the same in {sample_type_dir}"

        if len(image_files) < num_samples:
            if allow_too_few_samples:
                num_samples = len(image_files)
            else:
                raise ValueError(
                    f"| Not enough samples in {sample_type_dir}. Requested {num_samples}, but only {len(image_files)} available."
                )
        # sample the files
        sampled_indexes = torch.randperm(len(image_files))[:num_samples]
        sampled_image_files.extend([image_files[i] for i in sampled_indexes])
        sampled_mask_files.extend([mask_files[i] for i in sampled_indexes])

    return sampled_image_files, sampled_mask_files


def train_val_split(
    images: list[Path],
    masks: list[Path],
    val_split: float,
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """
    Split the training data into training and validation sets.

    Parameters
    ----------
    images : list[Path]
        List of paths to the image files.
    masks : list[Path]
        List of paths to the mask files.
    val_split : float
        Fraction of the data to use for validation.
    """

    assert len(images) == len(masks), "Number of images and masks must be the same"

    # shuffle the data
    indices = torch.randperm(len(images))
    images = [images[i] for i in indices]
    masks = [masks[i] for i in indices]

    # split the data into training and validation sets
    val_size = int(len(images) * val_split)
    val_images = images[:val_size]
    val_masks = masks[:val_size]
    train_images = images[val_size:]
    train_masks = masks[val_size:]

    return train_images, train_masks, val_images, val_masks


# def load_data(image_files: list[Path], mask_files: list[Path], vmin: float, vmax: float) -> TensorDataset:
#     """
#     Load the data from the image and mask files, preprocessing it.

#     The images and masks must be corresponding to each other, ie the ith image is for the ith mask.

#     Parameters
#     ----------
#     image_files : list[Path]
#         List of paths to the image files.
#     mask_files : list[Path]
#         List of paths to the mask files.

#     Returns
#     -------
#     TensorDataset
#         A TensorDataset containing the images and masks as tensors.
#     """

#     images = []
#     masks = []

#     for image_file, mask_file in zip(image_files, mask_files):
#         # load the image and mask
#         image = torch.from_numpy(np.load(image_file)).float()

#         assert image.ndim == 2, f"Image must be 2D (H, W), but got {image.ndim}D"
#         # add channel dim to the image tensor since it won't have one
#         image = image.unsqueeze(0)  # add channel dimension to the image tensor
#         mask = torch.from_numpy(np.load(mask_file).astype(bool)).float()
#         assert mask.ndim == 3, f"Mask must be 3D (C, H, W), but got {mask.ndim}D"

#         assert torch.unique(mask).tolist() == [0.0, 1.0], f"Mask must be binary (0 or 1), but got {torch.unique(mask)}"

#         # clip and normalize
#         image = torch.clamp(image, vmin, vmax)
#         image = (image - vmin) / (vmax - vmin)

#         images.append(image)
#         masks.append(mask)

#     # stack the images and masks into tensors
#     images = torch.stack(images)
#     masks = torch.stack(masks)

#     return TensorDataset(images, masks)


def get_loaders(
    images_paths: list[Path],
    masks_paths: list[Path],
    val_split: float,
    batch_size: int,
    vmin: float,
    vmax: float,
) -> tuple[DataLoader, DataLoader]:
    """
    Get the training and validation data loaders.

    Parameters
    ----------
    images_paths : list[Path]
        List of paths to the image files.
    masks_paths : list[Path]
        List of paths to the mask files.
    val_split : float
        Fraction of the data to use for validation.
    batch_size : int
        Batch size for the data loaders.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training and validation data loaders.
    """
    train_image_files, train_mask_files, val_image_files, val_mask_files = train_val_split(
        images_paths, masks_paths, val_split
    )

    # train_dataset = load_data(train_image_files, train_mask_files, vmin=vmin, vmax=vmax)
    # val_dataset = load_data(val_image_files, val_mask_files, vmin=vmin, vmax=vmax)
    train_dataset = SegmentationDataset(train_image_files, train_mask_files, vmin=vmin, vmax=vmax, augment=True)
    val_dataset = SegmentationDataset(val_image_files, val_mask_files, vmin=vmin, vmax=vmax, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
):
    # train the model for one epoch
    model.train()  # this sets the model to training mode
    # keep track of the running loss for this epoch
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, targets in progress_bar:
        images = images.to(device)
        targets = targets.to(device)

        # zero the exising gradients - so that they don't accumulate
        optimiser.zero_grad()

        # forward pass
        outputs = model(images)

        # compute the loss
        loss = criterion(outputs, targets)

        # backward pass & optimisation
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)  # multiply by batch size to get total loss for this batch
        progress_bar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)  # average loss for this epoch
    return epoch_loss


@torch.no_grad()  # disable gradient calculation for validation
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
):
    model.eval()  # set the model to evaluation mode
    running_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)  # multiply by batch size to get total loss for this batch

    epoch_loss = running_loss / len(dataloader.dataset)  # average loss for this epoch
    return epoch_loss


class SegmentationDataset(torch.utils.data.Dataset):
    """Custom dataset for augmenting the segmentation data."""

    def __init__(
        self,
        image_files: list[Path],
        mask_files: list[Path],
        vmin: float,
        vmax: float,
        augment: bool,
    ) -> None:
        """Initialise."""
        self.image_files = image_files
        self.mask_files = mask_files
        self.vmin = vmin
        self.vmax = vmax
        self.augment = augment

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset and augment if needed."""
        image = torch.from_numpy(np.load(self.image_files[index])).float()  # [H, W]
        mask = torch.from_numpy(np.load(self.mask_files[index]).astype(bool)).float()  # [C, H, W]

        # normalise the image
        image = torch.clamp(image, self.vmin, self.vmax)
        image = (image - self.vmin) / (self.vmax - self.vmin)
        # add channel dimension to the image tensor
        image = image.unsqueeze(0)  # [C, H, W]

        if self.augment:
            # horizontal flip
            if torch.rand(1).item() < 0.5:
                image = image.flip(-1)
                mask = mask.flip(-1)
            # vertical flip
            if torch.rand(1).item() < 0.5:
                image = image.flip(-2)
                mask = mask.flip(-2)
            rotation = torch.randint(0, 4, (1,)).item()
            image = torch.rot90(image, rotation, [-2, -1])
            mask = torch.rot90(mask, rotation, [-2, -1])

        return image, mask


def main():

    # Grab samples from directories for each sample type
    train_image_files, train_mask_files = sample_type_split(
        sample_type_dirs=[TRAINING_DATA_DIR / sample_type for sample_type in SAMPLE_TYPES],
        num_samples_per_type=NUM_SAMPLES_PER_TYPE,
        allow_too_few_samples=ALLOW_TOO_FEW_SAMPLES,
    )

    # get data loaders
    train_loader, val_loader = get_loaders(
        images_paths=train_image_files,
        masks_paths=train_mask_files,
        val_split=0.2,
        batch_size=BATCH_SIZE,
        vmin=-1.0,
        vmax=5.0,
    )

    # initialise model, loss function, and optimiser
    model = ResUNet(in_channels=1, out_channels=2).to(DEVICE)
    criterion = PermutationInvariantDiceLoss_2_channel()
    # use adam since using batchnorm and relu
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Gradually drop the learning rate if the validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")

    print("\n--- Starting training ---\n")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        # step the LR scheduler with the validation loss
        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
        )

        # save checkpoint if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model with val loss: {best_val_loss:.4f}")

    print("\n--- Training complete ---\n")

    # Load the best model and evaluate on the validation set
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    val_loss = validate(model, val_loader, criterion, DEVICE)
    print(f"Best model validation loss: {val_loss:.4f}")

    # Plot some predictions from the validation set
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # convert logits to probabilities

            # plot the image
            plt.figure(figsize=(12, 8))
            plt.subplot(1, 5, 1)
            plt.imshow(images[0, 0].cpu(), cmap="gray")
            plt.title("Input Image")
            plt.axis("off")

            # plot the target mask channels
            plt.subplot(1, 5, 2)
            plt.imshow(targets[0, 0].cpu(), cmap="gray")
            plt.title("Target Mask Channel 1")
            plt.axis("off")

            plt.subplot(1, 5, 3)
            plt.imshow(targets[0, 1].cpu(), cmap="gray")
            plt.title("Target Mask Channel 2")
            plt.axis("off")

            # plot the predicted mask channels
            plt.subplot(1, 5, 4)
            plt.imshow(outputs[0, 0].cpu(), cmap="gray")
            plt.title("Predicted Mask Channel 1")
            plt.axis("off")

            plt.subplot(1, 5, 5)
            plt.imshow(outputs[0, 1].cpu(), cmap="gray")
            plt.title("Predicted Mask Channel 2")
            plt.axis("off")

            plt.tight_layout()
            # save the figure
            # create a guid for the image based on index and epoch
            image_guid = f"epoch{epoch+1}_batch{i+1}"
            plt.savefig(PREDICTIONS_DIR / f"prediction_{image_guid}.png")


if __name__ == "__main__":
    main()
