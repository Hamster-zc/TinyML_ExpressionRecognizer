import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler


# 模型构建函数（面向嵌入式设备优化）
def build_esp32_model(input_shape=(48, 48, 3), num_classes=7):
    # 使用EfficientNetB0基础架构（无预训练权重）
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='avg'  # 直接使用全局平均池化
    )

    # 轻量级分类头设计
    model = models.Sequential([
        base_model,
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation='softmax')
    ])

    # 计算复杂度分析
    print("模型参数统计：")
    model.summary()
    
    return model

# 数据增强配置（内存高效型）
def create_datagen():
    return ImageDataGenerator(
        rescale=1./127.5 - 1.0,  # 归一化到[-1, 1]
        rotation_range=6,         # 减少旋转幅度
        width_shift_range=0.08,    # 微调平移范围
        height_shift_range=0.08,
        shear_range=0.08,          # 降低剪切强度
        zoom_range=0.08,
        horizontal_flip=True,
        fill_mode='constant',      # 使用恒定填充减少计算
        brightness_range=[0.9, 1.1] # 限制亮度调整范围
    )

# 训练可视化回调
def create_callbacks():
    return [
        TensorBoard(log_dir='./logs', histogram_freq=0, profile_batch=0),
        EarlyStopping(monitor='val_accuracy', patience=10, mode='max'),
        ModelCheckpoint('esp32_model.h5', monitor='val_accuracy', 
                       save_best_only=True, mode='max')
    ]

# 训练流程
def train_model():
    # 超参数配置
    BATCH_SIZE = 96      # 增大批次减少内存碎片
    EPOCHS = 80
    INPUT_SHAPE = (48, 48, 3)

    # 数据管道配置
    train_datagen = create_datagen()
    val_datagen = ImageDataGenerator(rescale=1./127.5 - 1.0)

    # 数据流配置（适配目录结构）
    train_generator = train_datagen.flow_from_directory(
        'Training',
        target_size=INPUT_SHAPE[:2],
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        'PublicTest',
        target_size=INPUT_SHAPE[:2],
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # 模型构建与编译
    model = build_esp32_model(INPUT_SHAPE)
    model.compile(
    optimizer=optimizers.Nadam(learning_rate=2e-3),  # 初始学习率
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

    # 模型训练
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        callbacks=create_callbacks(),
        verbose=2
    )

    # 训练过程可视化
    visualize_training(history)

def visualize_training(history):
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Set')
    plt.plot(history.history['val_accuracy'], label='Validation Set')
    plt.title('Accuracy Curve')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Set')
    plt.plot(history.history['val_loss'], label='Validation Set')
    plt.title('Loss Curve')
    plt.ylabel('Loss Value')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

if __name__ == "__main__":
    train_model()