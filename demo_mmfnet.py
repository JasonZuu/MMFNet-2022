import torch

from mmfnet import MMFNet


def demo_mmfnet():
    # Create an instance of MMFNet
    mmf_net = MMFNet()

    # Generate dummy data
    # Assuming the input image size required by InceptionResnetV1 is 3 x 224 x 224
    dummy_imgs = torch.randn(2, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 pixels
    dummy_X_struc = torch.randn(2, 5)  # Adjust the size according to the expected structure data size

    # Pass the dummy data through the network
    with torch.no_grad():  # Ensure gradients are not computed for test pass
        output = mmf_net(dummy_imgs, dummy_X_struc)

    # Print the output shape or other relevant information
    print("Output shape:", output.shape)
    print("Output:", output)


if __name__ == "__main__":
    demo_mmfnet()
