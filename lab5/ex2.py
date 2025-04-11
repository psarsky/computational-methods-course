import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from skimage import data, color
from skimage.transform import resize
from numpy.linalg import norm


def prepare_image():
    img = data.astronaut()
    img = img[50:550, 50:550]

    if len(img.shape) == 3:
        img = color.rgb2gray(img)

    img = resize(img, (512, 512), anti_aliasing=True)

    return img


def compress_image(img, k_values):
    U, sigma, VT = linalg.svd(img)

    compressed_images = {}

    for k in k_values:
        sigma_k = np.zeros_like(img)
        np.fill_diagonal(sigma_k[:k, :k], sigma[:k])

        img_compressed = U @ sigma_k @ VT

        compressed_images[k] = img_compressed

    return compressed_images


def compare_images(original_img, compressed_images, k_values):
    n = len(k_values)

    differences = {}
    norms = {}

    for k in k_values:
        differences[k] = original_img - compressed_images[k]
        norms[k] = norm(differences[k])

    _, axes = plt.subplots(2, n+1, figsize=(15, 8))

    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title('Oryginalny obraz')
    axes[0, 0].axis('off')

    axes[1, 0].axis('off')

    for i, k in enumerate(k_values):
        axes[0, i+1].imshow(compressed_images[k], cmap='gray')
        axes[0, i+1].set_title(f'k = {k}')
        axes[0, i+1].axis('off')

        axes[1, i+1].imshow(differences[k], cmap='gray')
        axes[1, i+1].set_title(f'||I-I_a|| = {norms[k]:.2f}')
        axes[1, i+1].axis('off')

    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, [norms[k] for k in k_values], 'o-')
    plt.title('Zależność ||I-I_a|| od k')
    plt.xlabel('k (liczba wartości osobliwych)')
    plt.ylabel('||I-I_a||')
    plt.grid(True)
    plt.show()


def analyze_compression(img, k_values):
    original_size = img.shape[0] * img.shape[1]

    for k in k_values:
        compressed_size = k * (img.shape[0] + img.shape[1] + 1)
        compression_ratio = original_size / compressed_size
        print(f"k = {k}: Współczynnik kompresji = {compression_ratio:.2f}x")


def main():
    k_values = [1, 5, 10, 20, 50, 100, 200]

    img = prepare_image()
    compressed_images = compress_image(img, k_values)
    compare_images(img, compressed_images, k_values)
    analyze_compression(img, k_values)


if __name__ == "__main__":
    main()
