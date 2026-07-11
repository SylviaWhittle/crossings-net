"""Test the model with dummy data"""

import torch
from src.loss import PermutationInvariantDiceLoss_2_channel
from src.resunet import ResUNet

if __name__ == "__main__":
    batch_size = 1
    # Simulate a batch of 4 heightmap images of size 256, 256 with one channel - NOT RGB
    dummy_input = torch.randn(batch_size, 1, 256, 256)
    # Simulate a batch of 4 heightmap images of size 256, 256 with two channels for the target
    dummy_target_1 = torch.randint(0, 2, (batch_size, 2, 256, 256)).float()
    # Flip the channels of the target to simulate a different permutation
    dummy_target_2 = torch.flip(dummy_target_1, dims=[1])

    # initialise netowrk and loss
    model = ResUNet(in_channels=1, out_channels=2)
    criterion = PermutationInvariantDiceLoss_2_channel()

    # forward
    outputs = model(dummy_input)
    print("Output shape:", outputs.shape)  # should be [batch_size, 2, 256, 256]

    # loss for the first target
    loss1 = criterion(outputs, dummy_target_1)
    print("Loss for target 1:", loss1)
    # loss for the second target
    loss2 = criterion(outputs, dummy_target_2)
    print("Loss for target 2:", loss2)

    print(f"match? {torch.allclose(loss1, loss2)}")
