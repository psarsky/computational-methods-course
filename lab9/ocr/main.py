"""Optical Character Recognition (OCR) module with tests including different fonts, noise levels, and rotations."""

import os
from collections import defaultdict
from difflib import SequenceMatcher

from PIL import ImageOps
from util import (IMG_DIR, CHARACTERS, correct_image_rotation, create_text_image,
                  denoise_image, load_font, load_image, read_text)
from vis import display_results


def ocr(image, font_name, font_size, confidence_threshold):
    """Main OCR function that processes an image and returns recognized text."""
    denoised_image = denoise_image(image)
    rotated_image = correct_image_rotation(denoised_image)
    inverted_image = ImageOps.invert(rotated_image)
    cropped_image = inverted_image.crop(inverted_image.getbbox())

    char_positions = read_text(
        cropped_image, font_name, font_size, confidence_threshold
    )

    line_groups = defaultdict(list)
    line_height = create_text_image("a", font_name, font_size).height

    for y_pos, x_pos, character in char_positions:
        matching_line = None
        for existing_line_y in line_groups.keys():
            if abs(y_pos - existing_line_y) < line_height:
                matching_line = existing_line_y
                break

        if matching_line is not None:
            line_groups[matching_line].append((x_pos, character))
        else:
            line_groups[y_pos] = [(x_pos, character)]

    text_lines = [char_list for _, char_list in sorted(line_groups.items())]

    reconstructed_text = ""
    space_width_threshold = load_font(font_name, font_size).getlength(" ") * 0.7

    for line_chars in text_lines:
        if not line_chars:
            continue

        sorted_line = sorted(line_chars)
        previous_x_position = sorted_line[0][0]

        for char_index, (x_position, character) in enumerate(sorted_line):
            if char_index > 0:
                char_width = create_text_image(character, font_name, font_size).width
                gap_size = x_position - previous_x_position - char_width
                if gap_size > space_width_threshold:
                    reconstructed_text += " "

            reconstructed_text += character
            previous_x_position = x_position

        reconstructed_text += "\n"

    char_counter = {c: reconstructed_text.count(c) for c in CHARACTERS}
    return reconstructed_text, char_counter


def main():
    """Main function to run OCR tests with different fonts, noise levels, and rotations."""
    font_size = 40
    fonts = ["arial", "times"]
    conf_levels = [0.89, 0.91, 0.93, 0.95, 0.97]
    noise_levels = [0, 15, 30, 50]
    rotation_angles = [0, 15, 30, 50]

    text = """abcdefghijklmnopqrstuvwxyz0123456789?!.,
lorem ipsum dolor sit amet, consectetur adipiscing elit.
proin pretium eros neque, a sodales mi pulvinar ut. 
donec placerat malesuada efficitur? suspendisse!"""

    results = []
    print_char_counts = True

    for font in fonts:
        print(f"\nNOISE TEST - {font}")
        print("=" * 40 + "\n")

        for noise in noise_levels:
            create_text_image(text, font, font_size, noise)
            image = load_image(
                os.path.join(IMG_DIR, f"sample-{font}-{noise}-0.png"),
                invert=False,
            )
            print(f"Font: {font}, noise: {noise}\n")
            print(f"Input text:\n{text}\n")
            print("Detected text:\n")

            for conf in conf_levels:
                detected_text, character_counts = ocr(image, font, font_size, conf)
                accuracy = (
                    SequenceMatcher(None, detected_text.strip(), text.strip()).ratio()
                    * 100
                )

                results.append(
                    {
                        "font": font,
                        "test_type": "noise",
                        "noise_level": noise,
                        "rotation_angle": 0,
                        "confidence": conf,
                        "accuracy": round(accuracy, 2),
                    }
                )

                print(detected_text.strip())
                print(f"Confidence: {conf}, Accuracy: {accuracy:.2f}%\n")
                if print_char_counts:
                    print_char_counts = False
                    print("Character counts (only for the first image, for display purposes):")
                    for char, count in character_counts.items():
                        print(f"{char}: {count}")
                    print()
            print("-" * 40 + "\n")

        print(f"\n\nROTATION TEST - {font}")
        print("=" * 40 + "\n")

        for rotation in rotation_angles:
            create_text_image(
                text, font, font_size, noise_level=0, rotation_angle=rotation
            )
            image = load_image(
                os.path.join(IMG_DIR, f"sample-{font}-0-{rotation}.png"),
                invert=False,
            )
            print(f"Font: {font}, rotation: {rotation}\n")
            print(f"Input text:\n{text}\n")
            print("Detected text:\n")

            for conf in conf_levels:
                detected_text, character_counts = ocr(image, font, font_size, conf)
                accuracy = (
                    SequenceMatcher(None, detected_text.strip(), text.strip()).ratio()
                    * 100
                )

                results.append(
                    {
                        "font": font,
                        "test_type": "rotation",
                        "noise_level": 0,
                        "rotation_angle": rotation,
                        "confidence": conf,
                        "accuracy": round(accuracy, 2),
                    }
                )

                print(f"Confidence: {conf}")
                print(detected_text.strip())
                print(f"Accuracy: {accuracy:.2f}%\n")
            print("-" * 40 + "\n")

    display_results(results)


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    main()
