#include <Arduino.h>

void setup() {
  // 使用ESP32-S3兼容的串口初始化方式
  Serial.begin(115200);
  
  // 添加足够延迟确保USB CDC初始化完成
  delay(3000); // 重要：至少2-3秒等待
  
  Serial.println("\n===== 串口测试固件 =====");
  Serial.printf("芯片型号: %s\n", ESP.getChipModel());
  Serial.printf("CPU核心数: %d\n", ESP.getChipCores());
  Serial.printf("CPU频率: %d MHz\n", ESP.getCpuFreqMHz());
  Serial.printf("可用内存: %d bytes\n", ESP.getFreeHeap());
  
  // 检查PSRAM
  if (psramFound()) {
    Serial.printf("可用PSRAM: %d bytes\n", ESP.getFreePsram());
  } else {
    Serial.println("PSRAM不可用");
  }
  
  Serial.println("测试输出: 1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZ");
  Serial.println("==========================================");
}

void loop() {
  // 仅保持程序运行
  static int counter = 0;
  if (counter % 10 == 0) {
    Serial.printf("系统运行中 [%d]...\n", counter);
  }
  counter++;
  delay(100);
}