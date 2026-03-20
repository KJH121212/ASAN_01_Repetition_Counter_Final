#!/bin/bash
#SBATCH -J tojihoo_yolo_convert
#SBATCH -t 7-00:00:00
#SBATCH -o /home/tojihoo/logs/%A.out
#SBATCH --mail-type END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND
#SBATCH --mail-user jihu6033@gmail.com
#SBATCH -p TitanRTX
#SBATCH --gpus 0

# ------------------------------------------------------------
# 환경 설정
# ------------------------------------------------------------
export HTTP_PROXY=http://192.168.45.108:3128
export HTTPS_PROXY=http://192.168.45.108:3128
export http_proxy=http://192.168.45.108:3128
export https_proxy=http://192.168.45.108:3128

DOCKER_IMAGE_NAME="tojihoo/sam:v1.1"
DOCKER_CONTAINER_NAME="tojihoo_yolo_convert"
DOCKERFILE_PATH="/mnt/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/jupyter/sam3/Dockerfile"
WORKSPACE_PATH="/mnt/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/jupyter/sam3/"
RANDOM_PORT=$(( (RANDOM % 101) + 8000 ))  # 8000~8100 사이 포트

# ------------------------------------------------------------
# Docker 이미지 빌드
# ------------------------------------------------------------
echo "[INFO] Building Docker image: ${DOCKER_IMAGE_NAME}"
docker build -t ${DOCKER_IMAGE_NAME} -f ${DOCKERFILE_PATH} ${WORKSPACE_PATH}
if [ $? -ne 0 ]; then
    echo "[❌ ERROR] Docker build failed."
    exit 1
fi

# ------------------------------------------------------------
# Docker 컨테이너 실행
# ------------------------------------------------------------
echo "[INFO] Running container: ${DOCKER_CONTAINER_NAME}"
docker run -it --rm --shm-size 1TB \
    --name "${DOCKER_CONTAINER_NAME}" \
    -p ${RANDOM_PORT}:${RANDOM_PORT} \
    -v /mnt:/workspace \
    -e HTTP_PROXY=${HTTP_PROXY} \
    -e HTTPS_PROXY=${HTTPS_PROXY} \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    ${DOCKER_IMAGE_NAME} \
    bash -c "
        cd /workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/runner && \
        python3 yolo_json_to_txt.py
    "
echo "[✅ DONE] yolo_json_to_txt.py finished."
