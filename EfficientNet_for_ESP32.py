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
from tensorflow_model_optimization.sparsity import keras as sparsity  # 新增剪枝库
from tensorflow_model_optimization.sparsity.keras import strip_pruning  # 新增
from tensorflow.keras.experimental import CosineDecayRestarts  # 新增余弦退火学习率

# 模型构建函数（面向嵌入式设备优化）
def build_esp32_model(input_shape=(48, 48, 1), num_classes=7,pruning=False,phase='base'):
    # 使用EfficientNetB0基础架构（无预训练权重）
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='avg',  # 直接使用全局平均池化
    )

    reg_config = {
        'base': {'act_l1':1e-6, 'act_l2':1e-5, 'kernel_l1':1e-5, 'kernel_l2':1e-4},
        'prune': {'act_l1':1e-7, 'act_l2':1e-6, 'kernel_l1':1e-6, 'kernel_l2':1e-5},
        'fine_tune': {'act_l1':0, 'act_l2':0, 'kernel_l1':0, 'kernel_l2':0}
    }[phase]
    # 轻量级分类头设计
    model = models.Sequential([
        base_model,
        layers.Dense(128, activation='swish', 
                     activity_regularizer=tf.keras.regularizers.l1_l2(l1=reg_config['act_l1'],
                                                                      l2=reg_config['act_l2']),  # 通道级正则化
                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_config['kernel_l1'],
                                                                    l2=reg_config['kernel_l2'])),  # 新增带正则化的中间层
        layers.BatchNormalization(),  # 新增BN层
        layers.Dropout(0.2),  
        layers.Dense(num_classes, activation='softmax')
    ])

    if pruning:
        pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(
                initial_sparsity=0.20,
                final_sparsity=0.45,
                begin_step=2000,
                end_step=15000,
                frequency=300
            )
        }
        # 仅对全连接层剪枝
        for i, layer in enumerate(model.layers):
            if isinstance(layer, layers.Dense):
                model.layers[i] = sparsity.prune_low_magnitude(layer, **pruning_params)
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
def create_callbacks(pruning=False):
    callbacks = [
        TensorBoard(log_dir='./logs', histogram_freq=0, profile_batch=0),
        EarlyStopping(monitor='val_accuracy', 
                      patience=15, 
                      min_delta=0.005,
                      mode='max',
                      restore_best_weights=True),
        ModelCheckpoint('esp32_model.h5', monitor='val_accuracy', 
                       save_best_only=True, mode='max')
    ]
    # 添加剪枝回调 Modified
    if pruning:
        callbacks.append(
            ModelCheckpoint('prune_checkpoint.h5', 
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                save_format='tf')
)
    
    return callbacks

# 混淆矩阵可视化（新增功能）
def plot_confusion_matrix(model, generator):
    y_true = generator.classes
    y_pred = model.predict(generator, steps=generator.samples // generator.batch_size + 1)
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
    EPOCHS = 120
    # 阶段分配调整为：
    # 基础训练：0-60轮 (50%)
    # 剪枝训练：60-90轮 (25%)
    # 微调训练：90-120轮 (25%)
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
    # 第一阶段：基础训练
    print("\n=== 基础训练阶段 ===")
    model = build_esp32_model(INPUT_SHAPE, pruning=False,phase='base')
    model.compile(
        optimizer=optimizers.Nadam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=int(EPOCHS*0.5),  # 50%训练时间用于基础训练
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        callbacks=create_callbacks(pruning=False),
        verbose=2
    )

    # 第二阶段：剪枝训练 
    print("\n=== 剪枝训练阶段 ===")
    pruned_model = build_esp32_model(INPUT_SHAPE, phase='prune',pruning=True)
    total_steps = (train_generator.samples // BATCH_SIZE) * int(EPOCHS*0.3)  # 计算总步数
    pruned_model.compile(
    optimizer=optimizers.Adam(
            learning_rate = CosineDecayRestarts(
            initial_learning_rate=3e-4,  # 基础学习率提升50%
            first_decay_steps=total_steps//3,
            t_mul=2.0,
            m_mul=0.7
        )
    ),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
    
    pruning_history = pruned_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=int(EPOCHS*0.3),  # 30%时间用于剪枝
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        callbacks=create_callbacks(pruning=True),
        verbose=2
    )

    # 第三阶段：微调训练 
    print("\n=== 微调阶段 ===")
    final_model = build_esp32_model(INPUT_SHAPE,pruning=False,phase='fine_tune') # 加载最佳剪枝模型
    final_model.load_weights('prune_checkpoint.h5')  # 仅加载权重，保留新结构
    final_model = strip_pruning(final_model)  # 移除剪枝包装
    final_model.compile(
        optimizer=optimizers.Nadam(learning_rate=2e-4,clipnorm=2.0),  # 新增梯度裁剪
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    fine_tune_history = final_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=int(EPOCHS*0.2),  # 20%时间用于微调
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        callbacks=create_callbacks(pruning=False),
        verbose=2
    )

    # 保存最终模型 Modified
    final_model.save('esp32_model_pruned.h5')

    # 合并历史记录 Modified
    full_history = {
        'accuracy': history.history['accuracy'] + pruning_history.history['accuracy'] + fine_tune_history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'] + pruning_history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'],
        'loss': history.history['loss'] + pruning_history.history['loss'] + fine_tune_history.history['loss'],
        'val_loss': history.history['val_loss'] + pruning_history.history['val_loss'] + fine_tune_history.history['val_loss']
    }

    # 训练过程可视化
    visualize_training(full_history)
    # 生成混淆矩阵
    plot_confusion_matrix(final_model, val_generator)

def visualize_training(history):
    plt.figure(figsize=(12, 5))
    
    # 计算阶段边界 Modified
    total_epochs = len(history['accuracy'])
    base_end = int(total_epochs * 0.5)   # 基础训练结束位置
    prune_end = base_end + int(total_epochs * 0.3)  # 剪枝训练结束位置
    
    # ------------------- 准确率曲线 -------------------
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Set')
    plt.plot(history['val_accuracy'], label='Validation Set')
    
    # 添加阶段分割线 Added
    plt.axvline(x=base_end, color='r', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(x=prune_end, color='g', linestyle='--', linewidth=1, alpha=0.7)
    
    # 添加阶段标签 Added
    plt.text(base_end//2, 0.1, 'Base Train', ha='center', va='center', 
            backgroundcolor='w', fontsize=9)
    plt.text(base_end + (prune_end-base_end)//2, 0.1, 'Prune', ha='center', 
            va='center', backgroundcolor='w', fontsize=9)
    plt.text(prune_end + (total_epochs-prune_end)//2, 0.1, 'Fine-tune', 
            ha='center', va='center', backgroundcolor='w', fontsize=9)
    
    plt.title('Accuracy Curve')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    # ------------------- 损失曲线 -------------------
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Set')
    plt.plot(history['val_loss'], label='Validation Set')
    
    # 添加阶段分割线 Added
    plt.axvline(x=base_end, color='r', linestyle='--', linewidth=1, alpha=0.7,
               label='Phase Transition')
    plt.axvline(x=prune_end, color='g', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.title('Loss Curve')
    plt.ylabel('Loss Value')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    
    # 添加全局图例说明 Added
    plt.figtext(0.5, 0.01, 
               f"Phase Division: | 0-{base_end} (Base) | {base_end}-{prune_end} (Prune) | {prune_end}-{total_epochs} (Fine-tune) |",
               ha='center', fontsize=9, color='gray')
    
    plt.savefig('training_metrics.png')
    plt.show()

if __name__ == "__main__":
    train_model()

    import gc
    gc.collect()
    tf.keras.backend.clear_session()