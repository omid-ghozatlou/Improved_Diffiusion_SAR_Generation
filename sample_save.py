import os
import numpy as np
import matplotlib.pyplot as plt

# Load the saved samples
samples = np.load('./data/samples/SAR_superres_128_to_256_cond_sigma_2000//samples_32x128x128x1.npz')['arr_0']

# Create a directory to save the PNG files
png_save_path = './data/samples/SAR_superres_128_to_256_cond_sigma_2000/png_samples/'
os.makedirs(png_save_path, exist_ok=True)

# Iterate over the samples and save each one as a PNG file
for i in range(samples.shape[0]):
    plt.figure(figsize=(4, 4))
    plt.imshow(samples[i, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(png_save_path, f'sample_{i+1}.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"Samples saved as PNG files in {png_save_path}")
