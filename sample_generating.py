import os
import numpy as np
import matplotlib.pyplot as plt
from sample import main_sample

if __name__ == "__main__":
    # Configuration parameters
    url_model = "./models_data/mnist_32_cond_sigma_100/ema_0.9999_35_vb=0.9274_mse=0.0013.pt"
    num_samples = 16
    sample_class = 1
    batch_size = 16
    use_ddim = False
    diffusion_steps = 500
    max_clip_val = 1
    plot = False
    base_save_path = './data/samples/Mnist_32_cond/'
    
    # Dynamically create the save path using sample_class
    url_save_path = os.path.join(base_save_path, str(sample_class))

    # Check model path
    if not os.path.exists(url_model):
        raise ValueError('The "url_model" given as input does not exist.')

    if not os.path.exists(url_save_path):
        os.makedirs(url_save_path, exist_ok=True)

    # Generate samples
    main_sample(
        model_path=url_model,
        clip_denoised=True,
        max_clip_val=max_clip_val,
        num_samples=num_samples,
        sample_class=sample_class,
        batch_size=batch_size,
        use_ddim=use_ddim,
        diffusion_steps=diffusion_steps,
        url_save_path=url_save_path,
        to_0_255=False,
        plot=plot
    )
    
    # Dynamically construct the file name for the saved samples
    samples_file_name = f'samples_{num_samples}x32x32x1.npz'
    samples_file_path = os.path.join(url_save_path, samples_file_name)

    # Load the saved samples
    samples = np.load(samples_file_path)['arr_0']
    
    # Create a directory to save the PNG files
    png_save_path = os.path.join(url_save_path, 'png_samples/')
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
