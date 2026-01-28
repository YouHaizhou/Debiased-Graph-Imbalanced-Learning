#!/bin/bash

# --- 配置信息 ---
DATASET="Cora"
EPOCHS=200
MU=1.0
LAM=0.1
LOG_DIR="experiments/logs"
# 获取当前时间戳作为日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$LOG_DIR/cora_comparison_$TIMESTAMP.log"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "==========================================================" | tee -a "$RESULT_FILE"
echo "开始节点分类实验对比 - 数据集: $DATASET" | tee -a "$RESULT_FILE"
echo "==========================================================" | tee -a "$RESULT_FILE"

# 1. 标准半监督
echo "Step 1/4: Running Standard..." | tee -a "$RESULT_FILE"
python main.py --dataset "$DATASET" --mode standard --epochs "$EPOCHS" >> "$RESULT_FILE" 2>&1

# 2. 不平衡-无权
echo "Step 2/4: Running Imbalanced..." | tee -a "$RESULT_FILE"
python main.py --dataset "$DATASET" --mode imbalanced --epochs "$EPOCHS" >> "$RESULT_FILE" 2>&1

# 3. 不平衡-加权
echo "Step 3/4: Running Weighted..." | tee -a "$RESULT_FILE"
python main.py --dataset "$DATASET" --mode weighted --epochs "$EPOCHS" >> "$RESULT_FILE" 2>&1

# 4. IceBerg+ (Ours)
echo "Step 4/4: Running IceBerg+..." | tee -a "$RESULT_FILE"
python main.py --dataset "$DATASET" --mode iceberg_plus --epochs "$EPOCHS" --mu "$MU" --lam "$LAM" >> "$RESULT_FILE" 2>&1

echo "==========================================================" | tee -a "$RESULT_FILE"
echo "实验完成，结果已保存至: $RESULT_FILE" | tee -a "$RESULT_FILE"