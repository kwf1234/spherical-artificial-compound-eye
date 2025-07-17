import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def sensor_raw_data_process():
    src_file = r'interpolation_list.npy'
    dest_file = r'data_real_time.npy'
    interpolation_list = np.load(src_file)
    center = np.array([654, 460])
    x_step = 23
    pixel_10_length = 175
    pixel_1_length = x_step * 9 + pixel_10_length
    pixels_per_sample = 10
    interpolation_position = []
    for j in range(60):
        degree = np.pi / 2 + np.pi / 30 * j
        for i in range(pixels_per_sample):
            length = pixel_1_length - i * x_step
            new_position = center + np.array([round(length * np.cos(degree)),
                                              round(length * np.sin(degree))])
            interpolation_position.append(new_position)
    data_matrix = np.zeros((600, 1280, 960))
    for i in range(600):
        for j in range(len(interpolation_position)):
            x = interpolation_position[j][0]
            y = interpolation_position[j][1]
            data_matrix[i][x][y] = interpolation_list[i][j]
    np.save(dest_file, np.array(data_matrix))


def sensor_data_visualization(npy_path=r"data_real_time.npy",
                              npy_vis_img_save_dir=r"vis_img"):
    threshold = 0.00000001  # current smaller than this value will be ignored
    frames = np.load(npy_path)
    for frame_index in range(frames.shape[0]):
        abc_i = frames[frame_index]
        plt.rcParams.update({'font.size': 20, 'font.family': 'Arial'})
        fig, ax1 = plt.subplots(1, 1, figsize=(12.8, 9.6))
        center_x, center_y = 640, 480
        width, height = 1280, 960
        half_width = width / 2
        half_height = height / 2
        rectangle = patches.Rectangle((center_x - half_width, center_y - half_height),
                                      width, height, linewidth=3, edgecolor='black', facecolor='none')
        ax1.add_patch(rectangle)
        y, x = np.indices(abc_i.shape)
        ax1.scatter(x, y, c='none', edgecolors='none', s=10, marker='*')
        non_zero_indices = np.where(np.abs(abc_i) > threshold)
        non_zero_values = abc_i[non_zero_indices]
        ax1.scatter(non_zero_indices[0], non_zero_indices[1],
                    c=non_zero_values, cmap='viridis', s=50, marker='o')

        ax1.axis('off')
        ax1.set_aspect('equal')
        output_path = os.path.join(npy_vis_img_save_dir, f"Img000{frame_index:03}.png")
        plt.savefig(output_path, format='png', dpi=200)
        plt.close(fig)
        image = Image.open(output_path)
        left = 645
        top = 638
        right = 1980
        bottom = 1637
        cropped_image = image.crop((left, top, right, bottom))
        resized_image = cropped_image.resize((1280, 960), Image.Resampling.LANCZOS)
        resized_image.save(output_path, format='PNG')


if __name__ == "__main__":
    print("=============== start =================")
    sensor_raw_data_process()
    sensor_data_visualization()
    print("=============== done =================")
