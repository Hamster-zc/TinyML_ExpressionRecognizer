# model_quantization.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import traceback

def representative_dataset():
    # 使用验证集作为代表数据集（约10%数据）
    datagen = ImageDataGenerator(
        rescale=1./127.5 ,
        preprocessing_function=lambda x: (x - 1) * (1 + np.random.uniform(-0.03,0.03)) # 保留亮度抖动
    )
    generator = datagen.flow_from_directory(
        'Training',
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=1,
        class_mode='categorical',
        shuffle=True
    )
    
    # 生成200个代表性样本
    for _ in range(500):
        img, _ = next(generator)
        yield [img.astype(np.float32)]

def quantize_model():
    # 加载训练好的浮点模型
    model = tf.keras.models.load_model('esp32_model_pruned.h5')
    
    # 转换器配置（全整数量化）
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.experimental_new_quantizer = True
    converter.experimental_enable_resource_variables = True
    
    # 执行量化
    try:
        tflite_quant_model = converter.convert()
        with open('esp32_model_quant.tflite', 'wb') as f:
            f.write(tflite_quant_model)
        print(f"量化完成！模型大小：{len(tflite_quant_model)/1024:.1f} KB")
    except Exception as e:
        print(f"量化失败：{str(e)}")

def verify_quantization():
    interpreter = tf.lite.Interpreter(model_path='esp32_model_quant.tflite')
    interpreter.allocate_tensors()
    
    # 获取输入输出层的完整详细信息
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # 确保获取正确的量化参数
    input_scale = input_details['quantization_parameters']['scales'][0]
    input_zero_point = input_details['quantization_parameters']['zero_points'][0]
    output_scale = output_details['quantization_parameters']['scales'][0]
    output_zero_point = output_details['quantization_parameters']['zero_points'][0]
    
    # 与训练完全一致的数据预处理
    datagen = ImageDataGenerator(rescale=1./127.5,
                                 preprocessing_function=lambda x:x - 1.0)  # 保持[-1,1]范围
    generator = datagen.flow_from_directory(
        'PublicTest',
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=1,
        class_mode='categorical',
        shuffle=False  # 必须关闭shuffle
    )
    
    correct = 0
    total = generator.samples
    
    try:
        # 使用更可靠的遍历方式
        for i in range(total):
            # 添加进度显示
            if i % 500 == 0:
                print(f"Processing sample {i}/{total}...")
            
            img, label = next(generator)
            
            # 输入数据量化处理
            quantized_img = np.clip(img / input_scale + input_zero_point, -128, 127)
            quantized_img = quantized_img.astype(input_details['dtype'])  # 自动匹配输入类型
            
            # 设置输入张量
            interpreter.set_tensor(input_details['index'], quantized_img)
            
            # 执行推理
            interpreter.invoke()
            
            # 获取并反量化输出
            output = interpreter.get_tensor(output_details['index'])
            output_float = (output.astype(np.float32) - output_zero_point) * output_scale
            
            # 统计正确率
            if np.argmax(output_float) == np.argmax(label):
                correct += 1
                
    except Exception as e:
        print(f"验证过程中发生错误：{str(e)}")
        print(f"最后处理的样本索引：{i}")
        traceback.print_exc()
        return
    
    # 添加详细统计信息
    class_names = list(generator.class_indices.keys())
    print("\n验证结果汇总：")
    print(f"总样本数：{total}")
    print(f"正确识别数：{correct}")
    print(f"准确率：{correct/total:.2%}")
    print("类别对应关系：", dict(zip(range(7), class_names)))

if __name__ == "__main__":
    quantize_model()
    verify_quantization()