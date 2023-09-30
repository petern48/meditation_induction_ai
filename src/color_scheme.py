import torch

RED = 0
GREEN = 1
BLUE = 2
WARM = 3
COOL = 4
HIGH_B_HIGH_A = 5
HIGH_B_LOW_A = 6
LOW_B_HIGH_A = 7
LOW_B_LOW_A = 8
COLOR_SCHEMES = {'warm': WARM, 
                 'cool': COOL, 
                 'high_body_high_activation': HIGH_B_HIGH_A, 
                 'high_body_low_activation': HIGH_B_LOW_A, 
                 'low_body_high_activation': LOW_B_HIGH_A, 
                 'low_body_low_activation': LOW_B_LOW_A
}

def apply_color_scheme(img, color_scheme):
    """Applies color scheme to all images in the images array"""
    try:  # convert string to int value
        color_scheme = COLOR_SCHEMES[color_scheme]
    except:
        raise Exception("Invalid Color Scheme. Exiting...")

    red_channel = img[:, :, :, RED]
    green_channel = img[:, :, :, GREEN]
    blue_channel = img[:, :, :, BLUE]

    if color_scheme == WARM:
        red_channel *= 1.3
        blue_channel *= 0.7

    elif color_scheme == COOL:
        red_channel *= 0.7
        blue_channel *= 1.3

    elif color_scheme == HIGH_B_HIGH_A:
        # target red and orange
        green_mask = (green_channel > red_channel)
        red_mask = ~ green_mask
        # red_mask = (red_channel > green_channel)
        # red_channel[green_mask] = green_channel[green_mask]
        # Scale until all red is more then green
        green_mask = (green_channel >= red_channel)
        red_channel[green_mask] *= 1.25
        green_channel[green_mask] *= 0.75

        # Remove the green where red is stronger
        red_channel[red_mask] *= 1.3
        green_channel[red_mask] *= 0.5

        blue_channel *= 0

    elif color_scheme == HIGH_B_LOW_A:
        # target deep blue or forest green
        # Get cells where blue > red and green
        # blue_mask = (blue_channel > green_channel)
        green_mask = (green_channel > blue_channel)
        blue_mask = ~ green_mask
        # Darken the colors
        green_channel[green_mask] *= 0.5
        blue_channel[green_mask] *= 0.1

        blue_channel[blue_mask] *= 0.5
        green_channel[blue_mask] *= 0.1

        red_channel *= 0

    elif color_scheme == LOW_B_HIGH_A:
        # target bold yellow or electric blue
        blue_mask = (blue_channel > red_channel) & (blue_channel > green_channel)
        img[blue_mask, BLUE] *= 1.3
        img[not blue_mask, RED] *= 1.3,
        img[not blue_mask, GREEN] *= 1.3

    # Already set c_dim to 1
    # elif color_scheme == LOW_B_LOW_A:  # instead just set c_dim to 1
    #     # target gray
    #     # set all the rgb values to the median rgb value
    #     median_rgb = torch.median(img, axis=3).values.to(torch.float32)
    #     img = median_rgb.unsqueeze(3).expand_as(img)

    else:
        raise Exception("Invalid Color Scheme. Exiting...")

    return img
