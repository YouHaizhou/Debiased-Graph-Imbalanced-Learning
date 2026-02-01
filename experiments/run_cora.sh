#!/bin/bash

export LANG=en_US.UTF-8
export PYTHONIOENCODING=utf-8

# 诊断单次运行（伪标签分布、pi_c、少数类 logit）：不加 >> "$RESULT_FILE"，直接看终端
#   python main.py --mode iceberg_plus --threshold_strategy dynamic_mean --diagnose
# 或: python main.py --mode iceberg_original --diagnose

# --- 配置信息 ---
DATASET="Cora"
EPOCHS=200
MU=1.0
LAM=0.1
LOG_DIR="experiments/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$LOG_DIR/cora_comparison_$TIMESTAMP.log"

mkdir -p "$LOG_DIR"

# 辅助函数：打印分割线
print_sep() {
    echo "==========================================================" | tee -a "$RESULT_FILE"
}

print_header() {
    echo ">>> CATEGORY: $1" | tee -a "$RESULT_FILE"
    echo "----------------------------------------------------------" | tee -a "$RESULT_FILE"
}

print_sep
echo "节点分类实验对比 - 数据集: $DATASET" | tee -a "$RESULT_FILE"
echo "开始时间: $(date)" | tee -a "$RESULT_FILE"
print_sep

# --- 第一类：Baseline Models (GCN Variants) ---
print_header "BASELINE MODELS (GCN)"
echo "[Strategy] Standard Semi-supervised" | tee -a "$RESULT_FILE"
python main.py --mode standard --epochs "$EPOCHS" >> "$RESULT_FILE" 2>&1

echo -e "\n[Strategy] Imbalanced (No Weighting)" | tee -a "$RESULT_FILE"
python main.py --mode imbalanced --epochs "$EPOCHS" >> "$RESULT_FILE" 2>&1

echo -e "\n[Strategy] Imbalanced (Weighted Loss)" | tee -a "$RESULT_FILE"
python main.py --mode weighted --epochs "$EPOCHS" >> "$RESULT_FILE" 2>&1
echo -e "----------------------------------------------------------\n" | tee -a "$RESULT_FILE"

# --- 第二类：IceBerg (Original Architecture) ---
print_header "ICEBERG (ORIGINAL)"
echo "[Strategy] Dynamic Mean Threshold (As per paper)" | tee -a "$RESULT_FILE"
python main.py --mode iceberg_original --epochs "$EPOCHS" --mu "$MU" --lam "$LAM" --beta 0 >> "$RESULT_FILE" 2>&1
echo -e "----------------------------------------------------------\n" | tee -a "$RESULT_FILE"

# --- 第三类：IceBerg+ (Our Baseline Framework) ---
print_header "ICEBERG+ (OUR FRAMEWORK)"
echo "[Sub-Strategy 1] Fixed Threshold (0.1)" | tee -a "$RESULT_FILE"
python main.py --mode iceberg_plus --threshold_strategy fixed --fixed_threshold 0.1 --diagnose>> "$RESULT_FILE" 2>&1

echo -e "\n[Sub-Strategy 2] Dynamic Mean" | tee -a "$RESULT_FILE"
python main.py --mode iceberg_plus --threshold_strategy dynamic_mean --diagnose>> "$RESULT_FILE" 2>&1

echo -e "\n[Sub-Strategy 3] Global Mean (Cora Statistics)" | tee -a "$RESULT_FILE"
python main.py --mode iceberg_plus --threshold_strategy global_mean --diagnose>> "$RESULT_FILE" 2>&1
echo -e "----------------------------------------------------------\n" | tee -a "$RESULT_FILE"

print_sep
echo "所有实验已完成！" | tee -a "$RESULT_FILE"
echo "日志文件: $RESULT_FILE" | tee -a "$RESULT_FILE"
print_sep