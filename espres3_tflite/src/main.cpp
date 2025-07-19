#include <Arduino.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_camera.h"
#include "FFat.h"
#include <esp_sleep.h>
#include <TFT_eSPI.h>
#include <SPI.h>

// ====================== LCD配置 ======================
TFT_eSPI tft = TFT_eSPI();

// 屏幕尺寸
#define LCD_WIDTH  240
#define LCD_HEIGHT 240

// LCD硬件引脚定义
#define TFT_RST  -1
#define TFT_BL   22

// ====================== 按键配置 ======================
#define BOOT_PIN 0
#define KEY0_PIN 23
#define KEY1_PIN 24
#define KEY2_PIN 25
#define KEY3_PIN 26

bool bootPressed = false;
bool key0Pressed = false;
bool key1Pressed = false;
bool key2Pressed = false;
bool key3Pressed = false;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

enum SystemMode {
  MODE_REALTIME_DETECTION,
  MODE_SINGLE_SHOT,
  MODE_PERFORMANCE_TEST
};
SystemMode currentMode = MODE_REALTIME_DETECTION;

// ====================== 摄像头配置 ======================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define PCLK_GPIO_NUM     45
#define SIOD_GPIO_NUM     39
#define SIOC_GPIO_NUM     38
#define Y2_GPIO_NUM        4  
#define Y3_GPIO_NUM        5  
#define Y4_GPIO_NUM        6  
#define Y5_GPIO_NUM        7  
#define Y6_GPIO_NUM       15  
#define Y7_GPIO_NUM       16  
#define Y8_GPIO_NUM       17  
#define Y9_GPIO_NUM       18  
#define VSYNC_GPIO_NUM    47
#define HREF_GPIO_NUM     48

// ====================== 模型配置 ======================
#define INPUT_WIDTH   48
#define INPUT_HEIGHT  48
#define INPUT_CHANNELS 1
#define INPUT_SIZE (INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS)
#define OUTPUT_SIZE   7
#define TENSOR_ARENA_SIZE 48 * 1024  // 48KB

const char* EMOTIONS[OUTPUT_SIZE] = {
  "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
};

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
uint8_t* tensor_arena = nullptr;
uint8_t* model_data = nullptr;
size_t model_size = 0;

// ====================== 函数声明 ======================
void preprocessImage(camera_fb_t* fb, uint8_t* dest);
camera_config_t getCameraConfig();
void checkKeys();
void handleBootButton();
void handleUserButtons();
void performExpressionDetection();
void saveCurrentImage();
void runModelPerformanceTest();
void initScreen();
void displayCameraPreview();
void displayResultScreen(const char* emotion, float confidence);
void drawEmotionBars(float* predictions);
void showErrorMessage(const char* message);
void showSystemInfo();

// ====================== 图像处理函数 ======================
void preprocessImage(camera_fb_t* fb, uint8_t* dest) {
  // 双线性下采样 96x96 → 48x48
  for (int y = 0; y < 48; y++) {
    for (int x = 0; x < 48; x++) {
      int src_x = x * 2;
      int src_y = y * 2;
      uint32_t sum = 
        fb->buf[src_y * 96 + src_x] +
        fb->buf[src_y * 96 + src_x + 1] +
        fb->buf[(src_y + 1) * 96 + src_x] +
        fb->buf[(src_y + 1) * 96 + src_x + 1];
      dest[y * 48 + x] = sum / 4;
    }
  }
}

// ====================== 摄像头配置函数 ======================
camera_config_t getCameraConfig() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  // 控制引脚
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM; 
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  
  return config;
}

// ====================== 屏幕初始化 ======================
void initScreen() {
  // 移除tft.reset()，改用硬件复位（若TFT_RST定义为有效引脚）：
  pinMode(TFT_RST, OUTPUT);
  digitalWrite(TFT_RST, LOW);
  delay(50);
  digitalWrite(TFT_RST, HIGH);
  delay(150);
     
  // 背光控制（IO1）
  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, HIGH); // 点亮背光
     
  // 初始化TFT
  tft.begin();
  tft.setRotation(TFT_ROTATION); // 使用配置文件中的旋转方向
  tft.fillScreen(TFT_BLACK);
  
  // 显示启动信息
  tft.setTextSize(2);
  tft.setTextDatum(MC_DATUM);
  tft.drawString("表情识别系统", LCD_WIDTH/2, LCD_HEIGHT/2 - 20);
  tft.setTextSize(1);
  tft.drawString("初始化中...", LCD_WIDTH/2, LCD_HEIGHT/2 + 10);
  
  delay(1000);
}

// ====================== LCD显示函数 ======================
void displayCameraPreview() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;
  
  const int previewWidth = 48;
  const int previewHeight = 48;
  const int startY = 20;
  
  tft.startWrite();
  for (int y = 0; y < 96; y += 2) {
    for (int x = 0; x < 96; x += 2) {
      uint8_t gray = fb->buf[y * 96 + x];
      uint16_t color = tft.color565(gray, gray, gray);
      tft.drawPixel(x/2 + (LCD_WIDTH - previewWidth)/2, 
                   y/2 + startY, 
                   color);
    }
  }
  tft.endWrite();
  
  esp_camera_fb_return(fb);
}

void displayResultScreen(const char* emotion, float confidence) {
  tft.fillScreen(TFT_BLACK);
  
  tft.setTextSize(2);
  tft.setTextColor(TFT_YELLOW);
  tft.drawString("表情识别", LCD_WIDTH/2, 15);
  
  tft.setTextSize(3);
  tft.setTextColor(TFT_GREEN);
  tft.drawString(emotion, LCD_WIDTH/2, LCD_HEIGHT/2 - 25);
  
  tft.setTextSize(2);
  tft.setTextColor(TFT_CYAN);
  char confStr[20];
  sprintf(confStr, "置信度: %.1f%%", confidence * 100);
  tft.drawString(confStr, LCD_WIDTH/2, LCD_HEIGHT/2);
}

void showErrorMessage(const char* message) {
  tft.fillScreen(TFT_RED);
  tft.setTextSize(2);
  tft.setTextColor(TFT_WHITE);
  tft.drawString("错误", LCD_WIDTH/2, LCD_HEIGHT/2 - 20);
  tft.setTextSize(1);
  tft.drawString(message, LCD_WIDTH/2, LCD_HEIGHT/2 + 10);
}

void drawEmotionBars(float* predictions) {
  const int barWidth = 18;
  const int spacing = 4;
  const int startX = 15;
  const int baseY = LCD_HEIGHT - 10;
  const int maxHeight = 50;
  
  tft.fillRect(0, baseY - maxHeight, LCD_WIDTH, maxHeight + 20, TFT_BLACK);
  
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    int barHeight = predictions[i] * maxHeight;
    int x = startX + i * (barWidth + spacing);
    
    tft.fillRect(x, baseY - barHeight, barWidth, barHeight, TFT_BLUE);
    tft.drawRect(x, baseY - barHeight, barWidth, barHeight, TFT_WHITE);
    
    tft.setTextSize(1);
    tft.setTextColor(TFT_WHITE);
    tft.drawString(String(EMOTIONS[i][0]), x + barWidth/2 - 3, baseY + 3);
  }
  
  tft.setTextSize(1);
  tft.setTextColor(TFT_YELLOW);
  tft.drawString("情感分布", LCD_WIDTH/2 - 15, baseY - maxHeight - 8);
}

// ====================== 按键处理函数 ======================
void checkKeys() {
  unsigned long currentTime = millis();
  
  bool currentBoot = (digitalRead(BOOT_PIN) == LOW);
  bool currentKey0 = (digitalRead(KEY0_PIN) == LOW);
  bool currentKey1 = (digitalRead(KEY1_PIN) == LOW);
  bool currentKey2 = (digitalRead(KEY2_PIN) == LOW);
  bool currentKey3 = (digitalRead(KEY3_PIN) == LOW);
  
  if (currentBoot != bootPressed) {
    if (currentTime - lastDebounceTime > debounceDelay) {
      bootPressed = currentBoot;
      lastDebounceTime = currentTime;
    }
  }
  
  key0Pressed = currentKey0;
  key1Pressed = currentKey1;
  key2Pressed = currentKey2;
  key3Pressed = currentKey3;
}

void handleBootButton() {
  static unsigned long pressStart = 0;
  static bool longHandled = false;
  
  if (bootPressed) {
    if (pressStart == 0) pressStart = millis();
    
    if (millis() - pressStart > 2000 && !longHandled) {
      Serial.println("长按BOOT: 进入烧录模式");
      tft.fillScreen(TFT_RED);
      tft.setTextColor(TFT_WHITE);
      tft.drawString("进入烧录模式", LCD_WIDTH/2, LCD_HEIGHT/2);
      delay(1000);
      esp_restart();
      longHandled = true;
    }
  } else {
    if (pressStart > 0 && !longHandled) {
      currentMode = (SystemMode)((currentMode + 1) % 3);
      Serial.printf("切换到模式: %d\n", currentMode);
      
      const char* modeNames[] = {"实时检测", "单次检测", "性能测试"};
      tft.fillScreen(TFT_BLUE);
      tft.setTextColor(TFT_WHITE);
      tft.drawString(modeNames[currentMode], LCD_WIDTH/2, LCD_HEIGHT/2);
      delay(500);
    }
    pressStart = 0;
    longHandled = false;
  }
}

void handleUserButtons() {
  static unsigned long lastPress = 0;
  
  if (millis() - lastPress < 300) return;
  
  if (key0Pressed) {
    lastPress = millis();
    Serial.println("KEY0按下: 保存图像");
    saveCurrentImage();
    tft.fillScreen(TFT_GREEN);
    tft.setTextColor(TFT_BLACK);
    tft.drawString("图像已保存", LCD_WIDTH/2, LCD_HEIGHT/2);
    delay(500);
  }
  
  if (key1Pressed) {
    lastPress = millis();
    Serial.println("KEY1按下: 性能测试");
    runModelPerformanceTest();
  }
  
  if (key2Pressed) {
    lastPress = millis();
    Serial.println("KEY2按下: 系统信息");
    showSystemInfo();
  }
  
  if (key3Pressed) {
    lastPress = millis();
    Serial.println("KEY3按下: 深度睡眠");
    tft.fillScreen(TFT_BLUE);
    tft.setTextColor(TFT_WHITE);
    tft.drawString("进入深度睡眠", LCD_WIDTH/2, LCD_HEIGHT/2);
    delay(1000);
    esp_deep_sleep_start();
  }
}

// ====================== 系统功能函数 ======================
void saveCurrentImage() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;
  
  static int imgCount = 0;
  String filename = "/capture_" + String(imgCount++) + ".raw";
  
  File file = FFat.open(filename, "wb");
  if (file) {
    file.write(fb->buf, fb->len);
    file.close();
    Serial.printf("已保存 %d 字节到 %s\n", fb->len, filename.c_str());
  } else {
    Serial.println("创建文件失败");
  }
  
  esp_camera_fb_return(fb);
}

void runModelPerformanceTest() {
  uint8_t testImg[INPUT_SIZE];
  for (int i = 0; i < INPUT_SIZE; i++) {
    testImg[i] = rand() % 256;
  }
  
  for (int i = 0; i < 5; i++) {
    memcpy(input->data.uint8, testImg, INPUT_SIZE);
    interpreter->Invoke();
  }
  
  const int RUNS = 50;
  unsigned long start = millis();
  for (int i = 0; i < RUNS; i++) {
    memcpy(input->data.uint8, testImg, INPUT_SIZE);
    interpreter->Invoke();
  }
  unsigned long duration = millis() - start;
  
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_YELLOW);
  tft.drawString("性能测试结果", LCD_WIDTH/2, 10);
  
  tft.setTextSize(1);
  tft.setTextColor(TFT_CYAN);
  tft.drawString("运行次数: " + String(RUNS), 20, 40);
  tft.drawString("总耗时: " + String(duration) + "ms", 20, 55);
  
  float avgTime = duration / (float)RUNS;
  tft.drawString("平均耗时: " + String(avgTime, 1) + "ms", 20, 70);
  
  float fps = 1000.0 / avgTime;
  tft.drawString("帧率: " + String(fps, 1) + "FPS", 20, 85);
  
  delay(3000);
}

void showSystemInfo() {
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_GREEN);
  tft.drawString("系统信息", LCD_WIDTH/2, 10);
  
  tft.setTextSize(1);
  tft.setTextColor(TFT_WHITE);
  
  int y = 30;
  tft.drawString("模型大小: " + String(model_size) + "字节", 10, y);
  y += 15;
  tft.drawString("输入尺寸: 48x48x1", 10, y);
  y += 15;
  tft.drawString("输出类别: " + String(OUTPUT_SIZE), 10, y);
  y += 15;
  
  const char* modes[] = {"实时检测", "单次检测", "性能测试"};
  tft.drawString("当前模式: " + String(modes[currentMode]), 10, y);
  
  delay(3000);
}

// ====================== 表情检测函数 ======================
void performExpressionDetection() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("摄像头捕获失败");
    return;
  }
  
  displayCameraPreview();
  
  uint8_t processedImg[INPUT_SIZE];
  preprocessImage(fb, processedImg);
  
  for (int i = 0; i < INPUT_SIZE; i++) {
    input->data.uint8[i] = processedImg[i];
  }
  
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("推理失败");
    esp_camera_fb_return(fb);
    return;
  }
  
  float* predictions = output->data.f;
  int maxIndex = 0;
  for (int i = 1; i < OUTPUT_SIZE; i++) {
    if (predictions[i] > predictions[maxIndex]) maxIndex = i;
  }
  
  displayResultScreen(EMOTIONS[maxIndex], predictions[maxIndex]);
  drawEmotionBars(predictions);
  
  Serial.print("检测到表情: ");
  Serial.print(EMOTIONS[maxIndex]);
  Serial.print(" (");
  Serial.print(predictions[maxIndex] * 100, 1);
  Serial.println("%)");
  
  esp_camera_fb_return(fb);
  delay(500);
}

// ====================== 主程序 ======================
void setup() {
  Serial.begin(115200);
  Serial.println("\n\n系统启动中...");
  
  // 检查文件系统空间
  size_t totalSpace = FFat.totalBytes();
  size_t usedSpace = FFat.usedBytes();
  size_t freeSpace = totalSpace - usedSpace;

  Serial.printf("FFat总空间: %.2f MB\n", totalSpace / (1024.0 * 1024));
  Serial.printf("FFat已用空间: %.2f MB\n", usedSpace / (1024.0 * 1024));
  Serial.printf("FFat可用空间: %.2f MB\n", freeSpace / (1024.0 * 1024));

  if (model_size > freeSpace) {
    showErrorMessage("空间不足加载模型!");
    Serial.printf("需要 %d 字节, 可用 %d 字节\n", model_size, freeSpace);
    while(1);
}
  // 1. 初始化PSRAM
  Serial.println("初始化PSRAM...");
  if (!psramInit()) {
    Serial.println("PSRAM初始化失败!");
    while(1) {
      Serial.println("系统因PSRAM失败暂停");
      delay(1000);
    }
  }
  Serial.println("PSRAM初始化成功");
  
  // 2. 初始化LCD屏幕
  Serial.println("初始化LCD...");
  initScreen();
  
  // 3. 初始化按键
  Serial.println("初始化按键...");
  pinMode(BOOT_PIN, INPUT_PULLUP);
  pinMode(KEY0_PIN, INPUT_PULLUP);
  pinMode(KEY1_PIN, INPUT_PULLUP);
  pinMode(KEY2_PIN, INPUT_PULLUP);
  pinMode(KEY3_PIN, INPUT_PULLUP);
  
  // 4. 初始化FFAT文件系统
  Serial.println("初始化FFAT...");
  if (!FFat.begin(true)) {
    showErrorMessage("FFAT初始化失败!");
    Serial.println("FFAT初始化失败!");
    while(1);
  }
  Serial.println("FFAT初始化成功");
  
  // 5. 加载模型文件
  Serial.println("加载模型...");
  const char* model_path = "/esp32_model_quant.tflite";
  File modelFile = FFat.open(model_path, "rb");
  if (!modelFile) {
    showErrorMessage("模型文件未找到!");
    Serial.printf("无法打开模型文件: %s\n", model_path);
    // 列出文件系统内容帮助调试
    Serial.println("文件系统内容:");
    File root = FFat.open("/");
    File file = root.openNextFile();
    while(file){
      Serial.printf("  %s (%d 字节)\n", file.name(), file.size());
      file = root.openNextFile();
      }
    while(1);
  }

  model_size = modelFile.size();
  Serial.printf("模型大小: %d 字节 (约%.2f MB)\n", model_size, model_size / (1024.0 * 1024));

  // 检查模型大小是否超过FFat分区
  if (model_size > 5 * 1024 * 1024) { // 5MB
    showErrorMessage("模型太大!");
    Serial.println("模型超过FFat分区大小!");
    while(1);
  }
  
  // 6. 分配PSRAM内存（优化分配策略）
  Serial.println("分配模型内存...");

  // 先分配模型数据
  model_data = (uint8_t*)ps_malloc(model_size);
  if (!model_data) {
    showErrorMessage("模型内存分配失败!");
    Serial.println("模型内存分配失败!");
    while(1);
  }

  // 然后分配Tensor Arena
  tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
  if (!tensor_arena) {
    showErrorMessage("张量内存分配失败!");
    Serial.println("张量内存分配失败!");
    free(model_data);
    while(1);
  }
  
  // 7. 读取模型数据
  if (modelFile.read(model_data, model_size) != model_size) {
    showErrorMessage("模型读取失败!");
    Serial.println("模型读取不完整!");
    while(1);
  }
  modelFile.close();
  Serial.println("模型加载成功");
  
  // 8. 初始化摄像头
  Serial.println("初始化摄像头...");
  camera_config_t config = getCameraConfig();
  esp_err_t camErr = esp_camera_init(&config);
  if (camErr != ESP_OK) {
    showErrorMessage("摄像头初始化失败!");
    Serial.printf("摄像头初始化失败! 错误代码: 0x%x\n", camErr);
    while(1);
  }
  Serial.println("摄像头初始化成功");
  
  // 9. 加载TensorFlow Lite模型
  Serial.println("加载TensorFlow模型...");
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    showErrorMessage("模型版本不匹配!");
    Serial.printf("模型版本不匹配: %d != %d\n", model->version(), TFLITE_SCHEMA_VERSION);
    while(1);
  }
  
  // 10. 添加模型所需算子
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddRelu();
  
  // 11. 构建解释器
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;
  
  // 12. 分配张量
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    showErrorMessage("张量分配失败!");
    Serial.println("张量分配失败!");
    while(1);
  }
  
  // 13. 获取输入输出张量指针
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  // 14. 系统就绪
  Serial.println("系统准备就绪");
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_GREEN);
  tft.drawString("系统就绪!", LCD_WIDTH/2, LCD_HEIGHT/2);
  
  Serial.printf("剩余内存: %d KB\n", ESP.getFreeHeap() / 1024);
  Serial.printf("PSRAM可用: %d KB\n", ESP.getFreePsram() / 1024);
  
  delay(1000);
}

void loop() {
  checkKeys();
  handleBootButton();
  handleUserButtons();
  
  switch(currentMode) {
    case MODE_REALTIME_DETECTION:
      performExpressionDetection();
      break;
      
    case MODE_SINGLE_SHOT:
      if (key0Pressed) {
        performExpressionDetection();
      } else {
        displayCameraPreview();
        delay(100);
      }
      break;
      
    case MODE_PERFORMANCE_TEST:
      displayCameraPreview();
      delay(100);
      break;
  }
}