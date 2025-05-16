#!/usr/bin/env bash
# ============================================================
# 3 信道 × 3 FGSM 组合；保存 log + table.txt
# ============================================================

# ---------- 公共超参 ----------
DATASET="imagenette"
MODEL="dae_vit_tiny"
DATA_DIR="./data/imagenette2-320"
BATCH=128
EPOCHS=50
SNR=25
# -----------------------------

CHANNELS=("awgn" "rayleigh" "rician")
FGSM_OPTS=("none" "0.1" "0.3")

LOG_DIR="exp_logs"
mkdir -p "$LOG_DIR"

for ch in "${CHANNELS[@]}"; do
  for eps in "${FGSM_OPTS[@]}"; do
    # FGSM 参数
    if [[ $eps == "none" ]]; then
      FGSM_ARGS=""
      tag="fgsm0"
    else
      FGSM_ARGS="--fgsm --fgsm-epsilon $eps"
      tag="fgsm${eps/./}"
    fi

    echo "===============  RUN  channel=$ch  $tag  ==============="
    base="${MODEL}_${ch}_${tag}"
    log_file="$LOG_DIR/${base}.log"

    # 训练并写日志
    python train.py \
      --dataset "$DATASET" \
      --model-name "$MODEL" \
      --data-dir "$DATA_DIR" \
      --batch-size $BATCH \
      --num-epochs $EPOCHS \
      --snr $SNR \
      --channel "$ch" \
      $FGSM_ARGS | tee "$log_file"

    # ---------- 抽取表格 ----------
    table_txt="$LOG_DIR/${base}_table.txt"
    awk '/┏/{flag=1} flag{print} /┗/{flag=0}' "$log_file" > "$table_txt"

    # ---------- 屏幕摘要 ----------
    echo "----  ${ch}  ${tag}  @SNR30dB  ----"
    grep -E "test_(loss|psnr|ssim)_snr30" "$table_txt"
    echo
  done
done

echo "全部 9 个实验完成！log 与 table.txt 已存至 $LOG_DIR/"