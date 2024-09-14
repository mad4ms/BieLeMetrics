import cv2
from PIL import Image


def mp4_to_gif(video_path, gif_path, fps=5, scale_percent=50, colors=256):
    vidcap = cv2.VideoCapture(video_path)
    frames = []

    success, image = vidcap.read()
    while success:
        # Resize frame
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        resized_image = cv2.resize(image, (width, height))

        # Convert from BGR (OpenCV format) to RGB (Pillow format)
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb).convert(
            "P", palette=Image.ADAPTIVE, colors=colors
        )  # Reduce colors
        frames.append(pil_image)  # Convert to PIL Image
        success, image = vidcap.read()

    # Save the frames as a GIF
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),  # Duration between frames
            loop=0,
            optimize=True,  # Optimize GIF
        )


# Example usage
mp4_to_gif(
    "assets/events/videos/demo.mp4", "assets/events/videos/demo.gif", fps=40
)
