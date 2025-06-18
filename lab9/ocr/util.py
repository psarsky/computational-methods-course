"""Module for OCR utility functions."""

import os
from collections import defaultdict
from functools import cmp_to_key

import cv2
import numpy as np
from numpy.fft import fft2, ifft2
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage import measure

CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789?!.,"
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")


def load_image(path, invert=True):
    """Load and optionally invert an image from file."""
    image = Image.open(path).convert("L")
    return ImageOps.invert(image) if invert else image


def load_font(name, size):
    """Create a font object from font name and size."""
    return ImageFont.truetype(f"fonts/{name}.ttf", size)


def create_text_image(
    text_content, font_name, font_size, noise_level=0, rotation_angle=0, inverted=False
):
    """Generate an image containing the specified text with optional transformations."""
    font = load_font(font_name, font_size)

    max_line_width = max(font.getlength(line) for line in text_content.split("\n"))
    img_width = int((max_line_width + 10) * 1.1)
    img_height = int(font_size * 1.5 * len(text_content.split("\n")))

    base_img = Image.new("L", (img_width, img_height), color="black")
    drawing_context = ImageDraw.Draw(base_img)
    drawing_context.text((10, 10), text_content, font=font, fill="white")

    content_bbox = base_img.getbbox()
    cropped_img = base_img.crop(content_bbox)

    rotated_img = cropped_img.rotate(rotation_angle, expand=True)
    inverted_img = ImageOps.invert(rotated_img)
    bordered_img = ImageOps.expand(inverted_img, border=1, fill="white")

    img_array = np.array(bordered_img)
    noisy_array = img_array + np.random.normal(0, noise_level, img_array.shape)
    clipped_array = np.clip(noisy_array, 0, 255)
    final_img = Image.fromarray(clipped_array.astype(np.uint8))

    result_img = ImageOps.invert(final_img) if inverted else final_img

    output_path = os.path.join(
        IMG_DIR, f"sample-{font_name}-{noise_level}-{rotation_angle}.png"
    )
    result_img.save(output_path)

    return result_img


def correct_image_rotation(image):
    """Automatically correct the rotation of text in an image."""
    img_array = np.array(ImageOps.invert(image.copy()))
    text_coordinates = np.column_stack(np.where(img_array > 0))
    center_point, dimensions, rotation_angle = cv2.minAreaRect(text_coordinates)

    if center_point[1] < dimensions[1]:
        rotation_angle = -rotation_angle
    else:
        rotation_angle = 90 - rotation_angle

    return image.rotate(
        rotation_angle, expand=True, fillcolor=255, resample=Image.BICUBIC
    )


def denoise_image(image):
    """Apply denoising filter to improve image quality."""
    img_array = np.array(image)
    denoised_array = cv2.fastNlMeansDenoising(img_array, None, 40, 7, 21)
    return Image.fromarray(denoised_array)


def combine_image_with_pattern(
    base_image, font_name, font_size, pattern_text=CHARACTERS, target_size=0
):
    """Add a pattern to the image for correlation analysis."""
    if target_size == 0:
        width, height = base_image.size
    else:
        width, height = target_size, target_size

    pattern_img = create_text_image(
        pattern_text,
        font_name,
        font_size,
        noise_level=0,
        rotation_angle=0,
        inverted=True,
    )

    combined_img = Image.new(
        "L", (max(width, pattern_img.width), height + pattern_img.height), 0
    )
    combined_img.paste(base_image, (0, 0))
    combined_img.paste(pattern_img, (0, height))

    return np.array(combined_img)


def cross_correlation(base_image, template_pattern):
    """Perform cross-correlation between image and pattern using FFT."""
    fft_image = fft2(base_image)
    fft_pattern = fft2(np.rot90(template_pattern, 2), np.array(base_image).shape)
    correlation_result = np.multiply(fft_image, fft_pattern)
    return np.real(ifft2(correlation_result))


def correlation_matrix(font_name, font_size):
    """Build a correlation matrix between all character pairs."""
    char_count = len(CHARACTERS)
    matrix = np.zeros((char_count, char_count))

    for i, first_char in enumerate(CHARACTERS):
        for j, second_char in enumerate(CHARACTERS):
            first_img = create_text_image(
                first_char, font_name, font_size, inverted=True
            )
            second_img = create_text_image(
                second_char, font_name, font_size, inverted=True
            )

            combined_img = combine_image_with_pattern(
                first_img, font_name, font_size, second_char
            )
            correlation_data = cross_correlation(combined_img, second_img)

            max_corr_region = correlation_data[
                : first_img.height, : second_img.width
            ].max()
            global_max_corr = correlation_data.max()
            matrix[i, j] = max_corr_region / global_max_corr

    return matrix


def recognition_order(font_name, font_size, confidence_threshold=0.99):
    """Determine optimal order for character recognition based on uniqueness."""
    char_frequency_map = defaultdict(list)

    background_img = np.array(
        create_text_image(CHARACTERS, font_name, font_size, inverted=True)
    )

    for character in CHARACTERS:
        char_pattern = np.array(
            create_text_image(character, font_name, font_size, inverted=True)
        )
        correlation_result = cross_correlation(background_img, char_pattern)

        correlation_result[
            correlation_result < np.max(correlation_result) * confidence_threshold
        ] = 0.0

        contour_count = len(
            measure.find_contours(
                correlation_result, confidence_threshold * np.max(correlation_result)
            )
        )
        char_frequency_map[contour_count].append(character)

    char_correlation_heatmap = correlation_matrix(font_name, font_size)
    order = []

    for bucket_key in sorted(char_frequency_map.keys()):
        sorted_chars = sorted(
            char_frequency_map[bucket_key],
            key=cmp_to_key(
                lambda char_y, char_x: char_correlation_heatmap[
                    CHARACTERS.index(char_x), CHARACTERS.index(char_y)
                ]
                - char_correlation_heatmap[
                    CHARACTERS.index(char_y), CHARACTERS.index(char_x)
                ]
            ),
        )
        order.extend(sorted_chars)

    return order


def read_text(image, font_name, font_size, confidence_level):
    """Extract text positions from image using pattern matching."""
    detected_positions = []
    char_recognition_order = recognition_order(font_name, font_size, confidence_level)
    processed_image = combine_image_with_pattern(
        image, font_name, font_size, CHARACTERS
    )

    for character in char_recognition_order:
        char_pattern = np.array(
            create_text_image(character, font_name, font_size, inverted=True)
        )
        correlation_data = cross_correlation(processed_image, char_pattern)

        correlation_data[
            correlation_data < np.max(correlation_data) * confidence_level
        ] = 0.0

        for row_idx, col_idx in np.argwhere(correlation_data != 0.0):
            for x_offset in range(1, char_pattern.shape[0]):
                for y_offset in range(1, char_pattern.shape[1]):
                    processed_image[row_idx - x_offset, col_idx - y_offset] = 0

            if row_idx <= image.height and col_idx <= image.width:
                detected_positions.append((row_idx, col_idx, character))

    return detected_positions
