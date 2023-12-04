import os
import imageio

def create_gif(image_folder, output_file, duration=0.5):
    """
    Create a GIF from a series of images.

    Parameters
    ----------
    image_folder : str
        The path to the folder containing the images.
    output_file : str
        The path and filename of the output GIF.
    duration : float, optional
        Duration of each frame in the GIF, by default 0.5 seconds.
    """
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.png'):
            file_path = os.path.join(image_folder, filename)
            images.append(imageio.imread(file_path))

    imageio.mimsave(output_file, images, duration=duration)

def create_mp4(image_folder, output_file, fps=10):
    """
    Create an MP4 video from a series of images.

    Parameters
    ----------
    image_folder : str
        The path to the folder containing the images.
    output_file : str
        The path and filename of the output MP4 video.
    fps : int, optional
        Frames per second in the output video, by default 10.
    """
    with imageio.get_writer(output_file, fps=fps) as writer:
        for filename in sorted(os.listdir(image_folder)):
            if filename.endswith('.png'):
                file_path = os.path.join(image_folder, filename)
                image = imageio.imread(file_path)
                writer.append_data(image)
