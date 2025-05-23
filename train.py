from ultralytics import YOLO
from ultralytics.utils import LOGGER  # 用于日志记录
import torch

if __name__ == '__main__':
    # --- 配置训练参数 ---
    model_yaml_path = 'ultralytics/cfg/models/v12/yolo12-FBRT.yaml'  # 您的模型结构YAML文件路径
    dataset_yaml_path = 'dataset/Visdrone/VisDrone.yaml'  # 您的数据集配置文件路径

    epochs_to_train = 100  # 训练轮数
    image_size = 640  # 图像尺寸
    batch_size = 8
    training_name = 'FBRT_yolo_custom_SGD_training'  # 本次训练的名称

    # 选择设备: '0' (第一个GPU), '0,1,2,3' (多个GPU), 或者 'cpu'
    device_to_use = '0' if torch.cuda.is_available() else 'cpu'

    try:
        # --- 加载模型配置 ---
        LOGGER.info(f"正在从模型配置文件加载模型结构: {model_yaml_path}")
        model = YOLO(model_yaml_path)
        LOGGER.info("模型结构加载完成。权重将随机初始化或根据YAML内部定义（如果有的话）。")

        # --- 开始训练 ---
        LOGGER.info(f"开始训练模型，数据集: {dataset_yaml_path}")
        LOGGER.info(
            f"训练参数: epochs={epochs_to_train}, imgsz={image_size}, batch={batch_size}, device='{device_to_use}', name='{training_name}'")
        LOGGER.info(
            f"优化器: SGD, 初始学习率 (lr0): 0.01, 动量: 0.937, 权重衰减: 0.0005")

        results = model.train(
            data=dataset_yaml_path,
            epochs=epochs_to_train,
            imgsz=image_size,
            batch=batch_size, # 使用更新后的batch_size
            device=device_to_use,
            name=training_name,
            # --- SGD特定参数 ---
            optimizer='SGD', # 指定优化器为SGD
            lr0=0.01, # 初始学习率
            momentum=0.937, # SGD动量
            weight_decay=0.0005, # 优化器权重衰减
            # --- 其他可选参数 ---
            # workers=8,
            # patience=50,
            # lrf=0.01, # 最终学习率 (lr0 * lrf) - 如果需要，可以取消注释并设置
            # resume=False,
            # plots=True,
        )

        LOGGER.info(f"训练完成！最好的模型权重保存在: {results.save_dir}/weights/best.pt")
        LOGGER.info(f"最终的模型权重保存在: {results.save_dir}/weights/last.pt")
        LOGGER.info(f"完整的训练结果保存在目录: {results.save_dir}")

    except Exception as e:
        LOGGER.error(f"训练过程中发生错误: {e}", exc_info=True)
        LOGGER.error("请检查：")
        LOGGER.error("1. 模型YAML和数据集YAML文件路径是否正确。")
        LOGGER.error("2. YAML文件内容是否符合Ultralytics的格式要求。")
        LOGGER.error("3. 数据集路径和文件是否都存在且可访问。")
        LOGGER.error("4. 所有自定义模块是否已正确实现并能被框架找到。")
        LOGGER.error("5. 您的硬件资源（如GPU显存）是否足够支持设定的批处理大小。")