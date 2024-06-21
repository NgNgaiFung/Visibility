import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

def threshold_image(image, threshold):
    thresholded_image = torch.where(image > threshold, 1, 0)
    if torch.all(thresholded_image.eq(1)):
        print("Thresholded image is white")
    return thresholded_image

image = Image.open("averaged_gray_level.jpg")
image = transforms.ToTensor()(image)

# Set the threshold value from the test list
threshold = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
threshold = [i/255 for i in threshold]

# Perform image thresholding
for i in threshold:
    thresholded_image = threshold_image(image, i)

    plt.subplot(5, 5, threshold.index(i)+1)
    plt.imshow(thresholded_image.squeeze().numpy(), cmap="gray")
    plt.axis("off")
    plt.title(f"T: {i*255}")

plt.tight_layout()
plt.savefig("adaptive_thresholding.png", dpi=300)
plt.show()

# # Display the original and thresholded images using matplotlib
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(image.squeeze().numpy())
# axes[0].set_title("Original Image")
# axes[0].axis("off")
# axes[1].imshow(thresholded_image.squeeze().numpy(), cmap="gray")
# axes[1].set_title("Thresholded Image (Threshold = {})".format(threshold))
# axes[1].axis("off")
# plt.show()