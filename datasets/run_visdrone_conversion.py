import os
from pathlib import Path
from PIL import Image  # 需要: pip install Pillow
from tqdm import tqdm  # 需要: pip install tqdm

# --- 配置部分 ---
# !!! 重要: 请将此路径修改为您的 VisDrone 数据集根目录的绝对路径 !!!
# 这个根目录应该包含像 "VisDrone2019-DET-train", "VisDrone2019-DET-val" 这样的子文件夹。
# 每个子文件夹内部应该有 "images" 和 "annotations" 两个子目录。
BASE_VISDRONE_DIR = Path(r"D:/A47z/A00l/yolo/yolov12/dataset/Visdrone/VisDrone_data")  # <--- 修改这里!

# 需要处理的数据集分割列表
SPLITS_TO_PROCESS = [
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
    "VisDrone2019-DET-test-dev"  # 如果您有测试集并且需要转换，请保留
]

# YOLO 期望的类别数量 (0索引)
# VisDrone 常用的10个目标类别，原始数据中通常用ID 1-10表示
# (0:ignored, 1:pedestrian, 2:people, 3:bicycle, 4:car, 5:van, 6:truck, 7:tricycle, 8:awning-tricycle, 9:bus, 10:motor, 11:others)
# 转换为YOLO后，这10个目标类别对应索引 0-9
NUM_CLASSES = 10


# --------------------

def convert_visdrone_bbox_to_yolo(image_width, image_height, visdrone_bbox):
    """
    将 VisDrone 边界框 (x_min, y_min, width, height) 转换为 YOLO 格式
    (x_center_norm, y_center_norm, width_norm, height_norm)。
    """
    x_min, y_min, w, h = visdrone_bbox

    if w <= 0 or h <= 0:
        # print(f"      跳过无效边界框 (宽度或高度 <= 0): {visdrone_bbox}")
        return None

    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0

    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = w / image_width
    height_norm = h / image_height

    # 将值限制在 [0.0, 1.0] 范围内，以处理轻微越界的框
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))

    # 再次检查归一化后的宽高是否有效
    if width_norm <= 0 or height_norm <= 0:
        # print(f"      跳过无效归一化边界框 (宽度或高度 <= 0): ({width_norm}, {height_norm})")
        return None

    return x_center_norm, y_center_norm, width_norm, height_norm


def process_single_split(split_dir_path: Path):
    """
    处理单个数据集分割 (例如 VisDrone2019-DET-train)。
    读取原始标注，转换为YOLO格式，并保存。
    """
    annotations_src_dir = split_dir_path / "annotations"
    images_src_dir = split_dir_path / "images"
    labels_output_dir = split_dir_path / "labels"

    print(f"\n正在处理分割: {split_dir_path.name}")
    print(f"  源标注文件夹: {annotations_src_dir}")
    print(f"  源图像文件夹: {images_src_dir}")
    print(f"  输出YOLO标签文件夹: {labels_output_dir}")

    if not annotations_src_dir.is_dir():
        print(f"  错误: 未找到标注文件夹: {annotations_src_dir}")
        return
    if not images_src_dir.is_dir():
        print(f"  错误: 未找到图像文件夹: {images_src_dir}")
        return

    labels_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  已创建/确认 'labels' 文件夹: {labels_output_dir}")

    original_annotation_files = sorted(list(annotations_src_dir.glob("*.txt")))
    if not original_annotation_files:
        print(f"  警告: 在 {annotations_src_dir} 中未找到 *.txt 原始标注文件。")
        return

    print(f"  找到 {len(original_annotation_files)} 个原始标注文件进行处理。")

    successful_label_files_created = 0
    total_valid_bboxes_converted = 0
    files_with_no_valid_bboxes = 0
    files_with_image_errors = 0
    malformed_annotation_lines = 0

    for ann_file_path in tqdm(original_annotation_files, desc=f"  转换中 {split_dir_path.name}"):
        image_file_stem = ann_file_path.stem
        yolo_label_lines = []

        img_path_jpg = images_src_dir / (image_file_stem + ".jpg")
        img_path_png = images_src_dir / (image_file_stem + ".png")  # VisDrone也可能有png格式

        image_path_to_open = None
        if img_path_jpg.exists():
            image_path_to_open = img_path_jpg
        elif img_path_png.exists():
            image_path_to_open = img_path_png

        if not image_path_to_open:
            # print(f"    警告: 未找到标注 {ann_file_path.name} 对应的图像文件 (尝试了 {img_path_jpg.name} 和 {img_path_png.name})。跳过。")
            files_with_image_errors += 1
            continue

        try:
            with Image.open(image_path_to_open) as img:
                img_width, img_height = img.size

            with open(ann_file_path, 'r', encoding='utf-8') as f_ann:
                for line_num, line in enumerate(f_ann):
                    parts = line.strip().split(',')
                    # VisDrone 格式: <bbox_left>,<bbox_top>,<width>,<height>,<score>,<object_category>,<truncation>,<occlusion>
                    if len(parts) < 6:  # 确保至少有6个部分来获取score和object_category
                        # print(f"    警告: 跳过格式错误的行 (列数不足) 于 {ann_file_path.name}, 行 {line_num + 1}: {line.strip()}")
                        malformed_annotation_lines += 1
                        continue

                    score = parts[4]
                    object_category_str = parts[5]

                    # 跳过 "ignored regions" (score == "0")
                    # 根据VisDrone文档，score为0表示忽略区域，1表示确定目标，2表示不确定目标。通常只用score为1的。
                    # 如果也想排除不确定目标，可以添加 `or score == "2"`
                    if score == "0":
                        continue

                    try:
                        original_class_id = int(object_category_str)
                        # VisDrone目标类别通常从1开始 (例如1:pedestrian, ..., 10:motor)
                        # 转换为0索引的YOLO类别ID
                        yolo_class_id = original_class_id - 1

                        if not (0 <= yolo_class_id < NUM_CLASSES):
                            # print(f"    警告: 原始类别ID {original_class_id} (YOLO ID {yolo_class_id}) 超出预期范围 [0, {NUM_CLASSES - 1}] 于 {ann_file_path.name}。行: {line.strip()}。跳过。")
                            continue

                        # <bbox_left>,<bbox_top>,<width>,<height>
                        # 确保这些值是数字
                        bbox_pixel_str = parts[0:4]
                        visdrone_bbox_pixel = tuple(map(int, map(float, bbox_pixel_str)))  # 先转float再转int，处理可能的"xxx.0"

                        yolo_bbox = convert_visdrone_bbox_to_yolo(img_width, img_height, visdrone_bbox_pixel)

                        if yolo_bbox:
                            xc, yc, w, h = yolo_bbox
                            yolo_label_lines.append(f"{yolo_class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                            total_valid_bboxes_converted += 1
                        # else:
                        # print(f"    信息: 边界框转换失败或无效于 {ann_file_path.name}: {line.strip()}")

                    except ValueError:
                        # print(f"    警告: 跳过类别ID或边界框坐标无法转换为数字的行于 {ann_file_path.name}, 行 {line_num + 1}: {line.strip()}")
                        malformed_annotation_lines += 1
                        continue

            # 即使没有有效目标，也为每个图像创建一个（可能为空的）标签文件
            output_label_file_path = labels_output_dir / (image_file_stem + ".txt")
            with open(output_label_file_path, 'w', encoding='utf-8') as f_label:
                f_label.writelines(yolo_label_lines)

            if yolo_label_lines:  # 如果此文件包含有效标签
                successful_label_files_created += 1
            else:
                files_with_no_valid_bboxes += 1


        except FileNotFoundError:
            print(
                f"    错误: 找不到图像文件对应标注: {ann_file_path.name}。已搜索: {image_path_to_open if image_path_to_open else '未定义'}")
            files_with_image_errors += 1
        except Exception as e:
            print(f"    处理标注文件 {ann_file_path.name} 时发生严重错误: {e}")
            files_with_image_errors += 1

    print(f"  '{split_dir_path.name}' 分割转换总结:")
    print(f"    处理的原始标注文件总数: {len(original_annotation_files)}")
    print(f"    成功创建/更新YOLO标签文件的数量 (即使为空): {len(original_annotation_files) - files_with_image_errors}")
    print(f"    其中，包含至少一个有效目标的标签文件数量: {successful_label_files_created}")
    print(f"    因图像无有效目标而生成空标签文件的数量: {files_with_no_valid_bboxes}")
    print(f"    转换得到的有效边界框总数: {total_valid_bboxes_converted}")
    print(f"    因格式错误或类别无效等原因跳过的单独标注行数: {malformed_annotation_lines}")  # 这个计数可能不完全准确，取决于跳过逻辑
    print(f"    因找不到对应图像或其他错误而未能处理的标注文件数: {files_with_image_errors}")


if __name__ == "__main__":
    print("开始 VisDrone 到 YOLO 标注转换过程...")
    print(f"VisDrone数据集根目录设置为: {BASE_VISDRONE_DIR}")

    if not BASE_VISDRONE_DIR.is_dir():
        print(f"致命错误: 未找到 VisDrone 数据集根目录: {BASE_VISDRONE_DIR}")
        print("请修改脚本顶部的 'BASE_VISDRONE_DIR' 变量为正确的路径。")
    else:
        for split_name in SPLITS_TO_PROCESS:
            current_split_path = BASE_VISDRONE_DIR / split_name
            if current_split_path.is_dir():
                process_single_split(current_split_path)
            else:
                print(f"\n警告: 数据集分割目录未找到，跳过: {current_split_path}")

    print("\n所有指定的数据集分割处理完毕。")
    print("请检查每个已处理分割文件夹中的 'labels' 子目录。")