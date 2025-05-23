from ultralytics import YOLO

# --- 1. 配置您的模型和数据集路径 ---
# 请将下面的路径替换成您自己的实际路径
MODEL_PATH = 'FBRT-yolo-final.pt'  # 您的 .pt 模型文件路径


DATASET_YAML_PATH = 'D:\A47z\A00l\yolo\yolov12\dataset\Visdrone\VisDrone.yaml' # 数据集配置文件 (非常重要!)
# 例如: 'D:/datasets/my_dataset/data.yaml' 或 '/home/user/datasets/coco128/coco128.yaml'

# --- 2. 主程序 ---
if __name__ == '__main__':
    print("开始加载模型...")
    # 加载您预训练的 YOLO 模型
    try:
        model = YOLO(MODEL_PATH)
        print(f"模型 '{MODEL_PATH}' 加载成功！")
    except Exception as e:
        print(f"加载模型 '{MODEL_PATH}' 失败: {e}")
        print("请检查：")
        print("1. 模型路径是否正确。")
        print("2. 模型文件是否损坏或与当前 Ultralytics 版本不兼容。")
        print("3. 如果您之前遇到 'AAttn' 相关的错误，请确保该问题已解决。")
        exit() # 如果模型加载失败，则退出程序

    print(f"\n开始在数据集 '{DATASET_YAML_PATH}' 上进行评估...")
    # 使用 model.val() 方法评估模型
    # data 参数需要指向您的数据集YAML配置文件
    # imgsz 参数可以指定评估时图像的尺寸，例如 640。如果您的模型是针对特定尺寸训练的，建议保持一致。
    # batch 参数可以指定批处理大小，根据您的GPU显存调整。
    # split 参数可以指定使用哪个数据集划分进行评估，默认为 'val' (验证集)，也可以是 'test' (测试集)，前提是您的YAML文件中有定义。
    try:
        metrics = model.val(data=DATASET_YAML_PATH,
                            imgsz=640,        # 您可以根据需要调整图像尺寸
                            batch=16,          # 如果显存不足，可以调小此值
                            split='val',      # 使用验证集进行评估
                            plots=True)       # 生成并保存评估图表 (如P-R曲线, 混淆矩阵等)

        print("\n--- 评估完成 ---")
        # metrics 对象包含了所有的评估结果
        # mAP@0.5 (IoU阈值为0.5时的平均精度)
        print(f"mAP@0.5 (Box): {metrics.box.map50:.4f}")
        # mAP@0.5:0.95 (IoU阈值从0.5到0.95，步长0.05，计算的平均mAP)
        print(f"mAP@0.5:0.95 (Box): {metrics.box.map:.4f}")
        # Precision (精确率)
        if metrics.box.mp is not None: # mp 可能不存在于非常早期的版本或特定情况下
            print(f"Precision (Box): {metrics.box.mp:.4f}")
        else:
            print("Precision (Box): 未直接获取到，请查看详细输出或保存的CSV文件。")
        # Recall (召回率)
        if metrics.box.mr is not None: # mr 可能不存在于非常早期的版本或特定情况下
            print(f"Recall (Box): {metrics.box.mr:.4f}")
        else:
            print("Recall (Box): 未直接获取到，请查看详细输出或保存的CSV文件。")

        print("\n评估结果和图表通常保存在项目的 'runs/detect/valX' 目录下 (X 是一个递增的数字)。")
        print("请检查该目录下的详细输出，例如：")
        print(" - confusion_matrix.png (混淆矩阵)")
        print(" - P_curve.png (Precision曲线)")
        print(" - R_curve.png (Recall曲线)")
        print(" - PR_curve.png (Precision-Recall曲线)")
        print(" - results.csv (包含各类别详细指标的CSV文件)")

    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        print("请检查：")
        print(f"1. 数据集配置文件路径 '{DATASET_YAML_PATH}' 是否正确。")
        print("2. 数据集格式是否符合 Ultralytics YOLO 的要求 (参考下面的YAML文件说明)。")
        print("3. 是否安装了 'pycocotools' (特别是对于COCO格式的数据集和指标计算): pip install pycocotools")
        print("4. 您的硬件资源（如GPU显存）是否足够。")