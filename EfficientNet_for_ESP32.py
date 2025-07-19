import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (TensorBoard, EarlyStopping, 
                                      ModelCheckpoint, ReduceLROnPlateau)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os 
from tensorflow_model_optimization.sparsity.keras import (
    strip_pruning,
    UpdatePruningStep,
    prune_low_magnitude,
    PolynomialDecay
)




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
        'prune': {'act_l1':1e-8, 'act_l2':1e-7, 'kernel_l1':0, 'kernel_l2':1e-6},
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
        total_steps = (28709 // 96) * 36  # 计算总步数
        pruning_params = {
            'pruning_schedule': PolynomialDecay(
                initial_sparsity=0.20,
                final_sparsity=0.75,
                begin_step=total_steps // 4,
                end_step=total_steps,
                frequency=100  
            )
        }
        # 仅对全连接层剪枝
        for i, layer in enumerate(model.layers):
            if isinstance(layer, layers.Dense) and hasattr(layer, 'kernel'):
                print(f" 剪枝层 {layer.name} ")  
                pruned_layer = prune_low_magnitude(layer, **pruning_params)
                pruned_layer._name = layer.name  # 保留原层名称
                model.layers[i] = pruned_layer

    model.summary()
    return model

# 数据增强配置（内存高效型）
def create_datagen():
    return ImageDataGenerator(
        rescale=1./127.5,  # 归一化到[-1, 1]
        rotation_range=6,         
        width_shift_range=0.08,    # 微调平移范围
        height_shift_range=0.05,
        shear_range=0.04,          # 降低剪切强度
        zoom_range=[0.9,1.1],
        channel_shift_range=10.0,   # 通道偏移
        horizontal_flip=True,
        fill_mode='constant',       
        preprocessing_function=lambda x: (x - 1.0) * (1 + np.random.uniform(-0.03,0.03)) # 亮度抖动
    )

# 训练可视化回调
def create_callbacks(pruning=False, phase='base'):
    """阶段化回调配置
    Args:
        pruning (bool): 是否剪枝阶段
        phase (str): 训练阶段标识 ('base', 'prune', 'fine_tune')
    """
    callbacks = [
        TensorBoard(
            log_dir=f'./logs/{phase}',  # 分阶段日志
            histogram_freq=0,
            profile_batch=0
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            min_delta=0.005,
            mode='max',
            restore_best_weights=True
        )
    ]
    
    # 分阶段配置模型保存
    if pruning:
        # 剪枝阶段保存完整模型结构
        callbacks += [
            ModelCheckpoint(
                'prune_checkpoint.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,  # 必须保存完整模型
                save_format='tf',        # 强制TF格式
                mode='max'
            )
        ]
        callbacks.append(UpdatePruningStep())  # 确保剪枝更新步骤被调用
    else:
        # 基础训练和微调阶段的保存路径区分
        filename = 'base_model.h5' if phase == 'base' else 'fine_tune_model.h5'
        callbacks.append(
            ModelCheckpoint(
                filename,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,  # 保存完整模型
                mode='max'
            )
        )
    
    return callbacks

# 训练流程
def train_model():
    # 超参数配置
    BATCH_SIZE = 96      # 增大批次减少内存碎片
    EPOCHS = 96
    INPUT_SHAPE = (48, 48, 1)

    # 数据管道配置
    train_datagen = create_datagen()
    val_datagen = ImageDataGenerator(rescale=1./127.5,
                                     preprocessing_function=lambda x: x - 1.0)

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
    base_epochs = 36
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=base_epochs,
        validation_data=val_generator,
        callbacks=create_callbacks(pruning=False,phase='base'),
        verbose=2
    )
    model.save('esp32_model.h5')  # 保存基础模型

    # 第二阶段：剪枝训练 
    print("\n=== 剪枝训练阶段 ===")
    pruned_model = build_esp32_model(INPUT_SHAPE, phase='prune',pruning=True)
    pruned_model.load_weights('esp32_model.h5', by_name=True)  # 加载基础模型权重
    # 添加剪枝回调
    pruning_callbacks = create_callbacks(pruning=True, phase='prune')
    # 添加额外的剪枝回调以确保执行
    if not any(isinstance(cb, UpdatePruningStep) for cb in pruning_callbacks):
        print("添加 UpdatePruningStep 回调...")
        pruning_callbacks.append(UpdatePruningStep())

    total_steps = (train_generator.samples // BATCH_SIZE) * 36
    pruned_model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
    )

    pruning_history = pruned_model.fit(
        train_generator,
        epochs=36, 
        initial_epoch=0,
        validation_data=val_generator,
        callbacks=pruning_callbacks,
        verbose=2
    )

    print("\n剪枝层稀疏度分析：")
    for layer in pruned_model.layers:
        if hasattr(layer, 'pruning'):
            weight = layer.weights[0].numpy()
            sparsity = 1.0 - np.count_nonzero(weight) / weight.size
            print(f"  ├─ {layer.name}: 实际稀疏度 {sparsity:.2%}")

    pruned_model.save('pruned_model.h5')  # 保存剪枝后的模型
    
    # 确保剪枝检查点存在
    if not os.path.exists('prune_checkpoint.h5'):
        print("警告：剪枝检查点未创建，保存最终模型作为备选")
        pruned_model.save('prune_checkpoint.h5')

    # 第三阶段：微调训练 
    print("\n=== 微调阶段 ===")
    if os.path.exists('prune_checkpoint.h5'):
        print("加载剪枝检查点模型...")
        try:
            final_model = tf.keras.models.load_model('prune_checkpoint.h5')
        except Exception as e:
            print(f"加载剪枝模型失败: {e}")
            print("尝试重建模型结构并加载权重...")
            final_model = build_esp32_model(INPUT_SHAPE, pruning=False, phase='fine_tune')
            final_model.load_weights('prune_checkpoint.h5', by_name=True)
    else:
        print("警告：找不到剪枝检查点，使用剪枝训练结束时的模型")
        final_model = pruned_model
    # 移除剪枝包装
    final_model = strip_pruning(final_model)

    final_model.compile(
        optimizer=optimizers.Nadam(learning_rate=1e-4,clipnorm=1.0),  # 新增梯度裁剪
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    fine_tune_history = final_model.fit(
        train_generator,
        epochs=24,
        initial_epoch=0,
        validation_data=val_generator,
        callbacks=create_callbacks(pruning=False),
        verbose=2
    )

    # 合并历史记录 
    full_history = {
        'accuracy': history.history['accuracy'] + pruning_history.history['accuracy'] + fine_tune_history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'] + pruning_history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'],
        'loss': history.history['loss'] + pruning_history.history['loss'] + fine_tune_history.history['loss'],
        'val_loss': history.history['val_loss'] + pruning_history.history['val_loss'] + fine_tune_history.history['val_loss']
    }

    # 保存最终模型 
    final_model.save('esp32_model_pruned.h5')
    # 训练过程可视化

    visualize_training(full_history)

def visualize_training(history):
    plt.figure(figsize=(12, 5))
    
    # 计算阶段边界 
    total_epochs = len(history['accuracy'])
    base_end = 36
    prune_end = 72

    # ------------------- 准确率曲线 -------------------
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Set')
    plt.plot(history['val_accuracy'], label='Validation Set')
    
    # 添加阶段分割线
    plt.axvline(x=base_end, color='r', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(x=prune_end, color='g', linestyle='--', linewidth=1, alpha=0.7)
    
    # 添加阶段标签
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
    
    # 添加阶段分割线
    plt.axvline(x=base_end, color='r', linestyle='--', linewidth=1, alpha=0.7,
               label='Phase Transition')
    plt.axvline(x=prune_end, color='g', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.title('Loss Curve')
    plt.ylabel('Loss Value')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    
    # 添加全局图例说明
    plt.figtext(0.5, 0.01, 
               f"Phase Division: | 0-{base_end} (Base) | {base_end}-{prune_end} (Prune) | {prune_end}-{total_epochs} (Fine-tune) |",
               ha='center', fontsize=9, color='gray')
    
    plt.savefig('training_metrics.png')
    plt.show()

if __name__ == "__main__":
    # 打印当前目录文件（调试用）
    print("当前目录文件:", os.listdir('.'))
    
    # 确保日志目录存在
    os.makedirs('./logs/base', exist_ok=True)
    os.makedirs('./logs/prune', exist_ok=True)
    os.makedirs('./logs/fine_tune', exist_ok=True)

    train_model()

    import gc
    gc.collect()
    tf.keras.backend.clear_session()