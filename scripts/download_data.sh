#!/usr/bin/env bash
# Download the CD-FSS evaluation sets (PATNet protocol) into $DATA (default: ./datasets).
#   FSS-1000   natural objects, 240 held-out test classes  (Google Drive, via gdown)
#   Deepglobe  satellite land cover, PATNet preprocessed  (Google Drive, via gdown)
#   ISIC-2018  dermoscopy, task 1                          (public S3)
#   Chest X-ray lungs (Shenzhen + Montgomery)              (Kaggle CLI, needs ~/.kaggle/kaggle.json)
# Run on a login node (compute nodes are often offline). Re-running skips what exists.
set -euo pipefail
DATA=${DATA:-${TABPFNSEG_DATA:-datasets}}
mkdir -p "$DATA" && cd "$DATA"
pip install -q gdown kaggle 2>/dev/null || true

if [ ! -d FSS-1000 ]; then
  gdown 16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI -O fss1000.zip
  unzip -q fss1000.zip && rm fss1000.zip
  # the archive unpacks to fewshot_data/ in some versions
  [ -d fewshot_data ] && mv fewshot_data FSS-1000 || true
fi

if [ ! -d Deepglobe ]; then
  gdown 12Dljy04maKIim3mZsR50CEOC3_ROZLCg -O deepglobe.zip
  unzip -q deepglobe.zip && rm deepglobe.zip
  echo "Deepglobe: expected layout Deepglobe/04_train_cat/<1..6>/test/{origin,groundtruth}; move folders if the archive differs."
fi

if [ ! -d ISIC ]; then
  mkdir -p ISIC && cd ISIC
  base=https://isic-archive.s3.amazonaws.com/challenges/2018
  for f in ISIC2018_Task1-2_Training_Input ISIC2018_Task1_Training_GroundTruth; do
    curl -L -o $f.zip $base/$f.zip && unzip -q $f.zip && rm $f.zip
  done
  cd ..   # classes are assigned from data/splits/isic/class_id.csv (PATNet) at load time
fi

if [ ! -d LungSegmentation ]; then
  kaggle datasets download -d nikhilpandey360/chest-xray-masks-and-labels -p lung_tmp --unzip
  # kaggle archive: Lung Segmentation/{CXR_png,masks}
  mv "lung_tmp/Lung Segmentation" LungSegmentation 2>/dev/null || mv lung_tmp/*/ LungSegmentation
  rm -rf lung_tmp
fi
# ---------------- underwater (manual: hosted on Google Drive / project pages, links change)
# SUIM  (Islam et al. IROS'20): https://irvlab.cs.umn.edu/resources/suim-dataset
#       -> unpack to SUIM/train_val/{images,masks} and SUIM/TEST/{images,masks}
# UWS   (Kabir et al., 'Few-shot segmentation and semantic segmentation for underwater imagery', IROS'23; link in the paper)
#       -> arrange as UWS/<class>/images/*.jpg + UWS/<class>/masks/<stem>.png
# UIIS (WaterMask, ICCV'23: github.com/LiamLian0727/WaterMask), USIS10K (arXiv 2406.06039), UIIS10K (UWSAM, arXiv 2505.15581)
#       -> <NAME>/annotations.json (+ <NAME>/images/); use the TEST split json
for d in SUIM UWS UIIS UIIS10K USIS10K; do [ -d "$d" ] && echo "found $d" || echo "missing $d (manual download, see comments)"; done
echo "done -> $(pwd)"; ls
