import os
import cv2
import shutil
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.interpolate import splprep, splev


def merge_close_points(cluster, distance_threshold):
    if len(cluster) < 2:
        return cluster
    distances = squareform(pdist(cluster))
    merged = np.full(cluster.shape[0], False)
    new_points = []
    for i in range(len(cluster)):
        if merged[i]:
            continue
        close_points = np.where((distances[i] < distance_threshold) & (distances[i] > 0))[0]
        if len(close_points) > 0:
            points_to_merge = cluster[close_points]
            points_to_merge = np.vstack((points_to_merge, cluster[i]))
            mean_point = np.mean(points_to_merge, axis=0).astype(int)
            merged[close_points] = True
            merged[i] = True
            new_points.append(mean_point)
        else:
            new_points.append(cluster[i])
    return np.array(new_points)


def order_points(points, center):
    ordered_points = []
    current_point = min(points, key=lambda p: euclidean(p, center))
    ordered_points.append(current_point)
    points = points.tolist()
    points.remove(current_point.tolist())
    while points:
        next_point = min(points, key=lambda p: euclidean(p, current_point))
        ordered_points.append(next_point)
        points.remove(next_point)
        current_point = next_point
    return np.array(ordered_points)


def rotate_circle_area(image, angle, center, radius):
    mask = np.zeros_like(image)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    circle_area = cv2.bitwise_and(image, mask)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(circle_area, M, (image.shape[1], image.shape[0]))
    inverse_mask = cv2.bitwise_not(mask)
    image = cv2.bitwise_and(image, inverse_mask)
    image = cv2.add(image, rotated_image)
    return image


def is_within_circle(point, center, radius):
    return (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2 < radius ** 2


def predict_by_sensor_data():
    input_folder = r'vis_img'
    output_name = r'predict'
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):
            if file_name.endswith('.txt') or file_name.endswith('_new.jpg'):
                os.remove(file_path)
    model = YOLO(r'best.pt')
    model.predict(source=input_folder, imgsz=1024, max_det=20, save_txt=True, project=input_folder, name=output_name, exist_ok=True, verbose=False)
    labels_folder = os.path.join(input_folder, output_name, 'labels')
    for file_name in os.listdir(labels_folder):
        src_path = os.path.join(labels_folder, file_name)
        dst_path = os.path.join(input_folder, file_name)
        shutil.move(src_path, dst_path)
    os.rmdir(labels_folder)
    os.rmdir(os.path.join(input_folder, output_name))


def read_label_and_fit_curve(directory=r"vis_img",
                             background_path=r"background_img.jpg",
                             distance_threshold=38):
    jpg_files = [f for f in os.listdir(directory) if f.endswith('.jpg') and not f.endswith('_new.jpg') and not f.endswith('_new_with_GT.jpg')]
    for index, jpg_file in enumerate(jpg_files):
        label_file = os.path.splitext(jpg_file)[0] + '.txt'
        label_path = os.path.join(directory, label_file)
        output_file = os.path.splitext(jpg_file)[0] + '_new.jpg'
        output_path = os.path.join(directory, output_file)
        if not os.path.exists(label_path):
            print(f"Label file {label_file} not found for image {jpg_file}")
            continue
        background_img = cv2.imread(background_path)
        if background_img is None:
            print(f"Background image {background_path} not found.")
            continue
        W, H = 1280, 960
        background_img = cv2.resize(background_img, (W, H))
        if index > 0:
            angle = 6 * index
            center = (630, 510)
            radius = 139
            background_img = rotate_circle_area(background_img, angle, center, radius)
        pixel_indices = []
        with open(label_path, 'r') as f:
            for line in f:
                _, x, y, _, _ = map(float, line.strip().split())
                pixel_x = int(x * W)
                pixel_y = int(y * H)
                pixel_indices.append([pixel_x, pixel_y])
        pixel_indices = np.array(pixel_indices)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(pixel_indices)
        labels = kmeans.labels_
        cluster_1 = pixel_indices[labels == 0]
        cluster_2 = pixel_indices[labels == 1]
        cluster_1 = merge_close_points(cluster_1, distance_threshold)
        cluster_2 = merge_close_points(cluster_2, distance_threshold)
        center = (W // 2, H // 2)
        cluster_1 = order_points(cluster_1, center)
        cluster_2 = order_points(cluster_2, center)
        pure_white = (255, 255, 255)

        def draw_spline_curve(cluster, color, thickness=2.5):
            if len(cluster) > 2:
                x = cluster[:, 0]
                y = cluster[:, 1] + 10
                tck, u = splprep([x, y], s=0, k=2)
                new_points = 500
                new_u = np.linspace(0, 1, new_points)
                new_skeleton = splev(new_u, tck)
                for i in range(len(new_skeleton[0]) - 1):
                    pt1 = (int(new_skeleton[0][i]), int(new_skeleton[1][i]))
                    pt2 = (int(new_skeleton[0][i + 1]), int(new_skeleton[1][i + 1]))
                    if not is_within_circle(pt1, (630, 510), 139) and not is_within_circle(pt2, (630, 510), 139):
                        cv2.line(background_img, pt1, pt2, color, thickness)

        draw_spline_curve(cluster_1, pure_white, thickness=25)
        draw_spline_curve(cluster_2, pure_white, thickness=25)
        background_img = cv2.GaussianBlur(background_img, (5, 5), 0)
        result_img = Image.fromarray(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB))
        result_img.save(output_path)


def generate_video(fps=4,
                   do_type="_new.jpg",
                   img_path=r"vis_img",
                   out_video=r'out.mp4'):
    to_imgs = [i for i in os.listdir(img_path) if i.endswith(do_type)]
    imgs = sorted(to_imgs)
    slow_down_factor = 2000 / fps
    text = f"Slowed Down by {slow_down_factor:.0f} Times"
    print(text)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = out_video
    out = cv2.VideoWriter(video_path, fourcc, fps, (1280, 960))
    for img_name in imgs:
        img_full_path = os.path.join(img_path, img_name)
        frame = cv2.imread(img_full_path)
        if frame is None:
            continue

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = frame.shape[1] - text_size[0] - 10  # 10 pixels from the right edge
        text_y = 80  # 20 pixels from the top
        # Put text on the frame
        # cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)
        # Write the frame to the video file
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()


def do_reconstruction():
    predict_by_sensor_data()
    read_label_and_fit_curve()
    generate_video()


if __name__ == "__main__":
    print("=============== start =================")
    do_reconstruction()
    print("=============== done =================")
