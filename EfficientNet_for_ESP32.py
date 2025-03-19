import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (TensorBoard, EarlyStopping, 
                                      ModelCheckpoint, ReduceLROnPlateau)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os


# 模型构建函数（面向嵌入式设备优化）
def build_esp32_model(input_shape=(48, 48, 1), num_classes=7):
    # 使用EfficientNetB0基础架构（无预训练权重）
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='avg',  # 直接使用全局平均池化
    )

    # 轻量级分类头设计
    model = models.Sequential([
        base_model,
        layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001,0.001)),  # 新增带正则化的中间层
        layers.BatchNormalization(),  # 新增BN层
        layers.Dropout(0.2),  
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
        rotation_range=6,         
        width_shift_range=0.08,    # 微调平移范围
        height_shift_range=0.05,
        shear_range=0.04,          # 降低剪切强度
        zoom_range=[0.9,1.1],
        channel_shift_range=10.0,   # 通道偏移
        horizontal_flip=True,
        fill_mode='constant',       
        preprocessing_function=lambda x: x * (1 + np.random.uniform(-0.03,0.03)) # 亮度抖动

    )

# 训练可视化回调
def create_callbacks():
    return [
        TensorBoard(log_dir='./logs', histogram_freq=0, profile_batch=0),
        ReduceLROnPlateau(
            monitor='val_accuracy',  # 监控验证精度
            factor=0.5,              
            patience=6,              # 6轮无提升即调整
            mode='max',
            min_lr=1e-5
        ),
        EarlyStopping(monitor='val_accuracy', patience=20, mode='max',restore_best_weights=True),
        ModelCheckpoint('esp32_model.h5', monitor='val_accuracy', 
                       save_best_only=True, mode='max')
    ]

# 混淆矩阵可视化（新增功能）
def plot_confusion_matrix(model, generator):
    y_true = generator.classes
    y_pred = model.predict(generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred_classes)
    class_names = list(generator.class_indices.keys())
    
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

# 训练流程
def train_model():
    # 超参数配置
    BATCH_SIZE = 96      # 增大批次减少内存碎片
    EPOCHS = 60
    INPUT_SHAPE = (48, 48, 1)

    # 数据管道配置
    train_datagen = create_datagen()
    val_datagen = ImageDataGenerator(rescale=1./127.5 - 1.0)

    # 数据流配置（适配目录结构）
    train_generator = train_datagen.flow_from_directory(
        'Training',
        target_size=INPUT_SHAPE[:2],
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        'PublicTest',
        target_size=INPUT_SHAPE[:2],
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # 模型构建与编译
    model = build_esp32_model(INPUT_SHAPE)

    # 新增权重加载代码
    try:
        model.load_weights('esp32_model.h5')
        print("成功加载已有权重！")
    # 调整初始学习率为上次训练结束时的值（可选）
        initial_lr = 2e-3 * (0.96 ** (EPOCHS // 5)) 
    except Exception as e:
        print(f"未找到权重文件，将从头开始训练。错误信息：{str(e)}")
    model.compile(
    optimizer=optimizers.Nadam(learning_rate=8e-4),  # 初始学习率
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
    # 生成混淆矩阵
    plot_confusion_matrix(model, val_generator)

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