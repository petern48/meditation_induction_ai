RED = 0
GREEN = 1
BLUE = 2


def apply_color_scheme(img, color_scheme):
    """Applies color scheme to all images in the images array"""

    red_channel = img[:, :, :, RED]
    green_channel = img[:, :, :, GREEN]
    blue_channel = img[:, :, :, BLUE]

    if color_scheme == 'warm':
        red_channel *= 1.3
        blue_channel *= 0.7

    elif color_scheme == 'cool':
        red_channel *= 0.7
        blue_channel *= 1.3

    elif color_scheme == 'red-orange':
        # target red and orange
        green_mask = (green_channel >= red_channel)
        red_mask = ~ green_mask
        red_channel[red_channel < 30] = 30
        red_channel *= 1.5
        # Turn green to half of red to make orange
        green_channel[green_mask] = red_channel[green_mask] * 0.5

        # Remove the green to highlight the red where red stronger
        green_channel[red_mask] = 0

        blue_channel.fill_(0)

    elif color_scheme == 'blue-green':
        # target deep blue or forest green
        green_mask = (green_channel >= blue_channel)
        blue_mask = ~ green_mask
        # Darken the colors
        green_channel[green_mask] *= 0.2
        blue_channel[green_mask] = 0

        blue_channel[blue_mask] *= 0.4
        green_channel[blue_mask] = 0

        red_channel.fill_(0)

    elif color_scheme == 'blue-yellow':
        # target electric blue and bold yellow
        blue_mask = (blue_channel > green_channel)
        green_mask = ~blue_mask

        red_channel[red_channel < 80] = 80
        green_channel[green_channel < 80] = 80

        # Electric Blue: Set blue and green equal
        blue_channel[blue_mask] = green_channel[blue_mask]
        red_channel[blue_mask] = 0

        # Places with more green turn to yellow
        green_channel[green_mask] = red_channel[green_mask]
        blue_channel[green_mask] = 0

    else:
        raise Exception("Invalid Color Scheme. Exiting...")

    return img
