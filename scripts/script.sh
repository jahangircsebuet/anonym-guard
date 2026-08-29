PROJECT_ROOT="/path/to/guardbreach-artifact"

CUDA_VISIBLE_DEVICES=3 nohup "${PROJECT_ROOT}/scripts/classifier.sh" \
  >> "${PROJECT_ROOT}/logs/classifier.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup "${PROJECT_ROOT}/scripts/results.sh" \
  >> "${PROJECT_ROOT}/logs/results.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup "${PROJECT_ROOT}/scripts/router.sh" \
  >> "${PROJECT_ROOT}/logs/router.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup "${PROJECT_ROOT}/scripts/data.sh" \
  >> "${PROJECT_ROOT}/logs/data.log" 2>&1 &