# Kế Hoạch Hoàn Thành Lab MLOps: Từ Thực Nghiệm Đến CI/CD (Phiên bản AWS)

Tài liệu này là hướng dẫn từng bước chi tiết giúp bạn hoàn thành bài lab MLOps dựa trên cấu trúc thư mục và yêu cầu của bài toán, sử dụng nền tảng **AWS (Amazon Web Services)**.

---

## Giai Đoạn 0: Chuẩn Bị & Thiết Lập Ban Đầu

### 1. Chuẩn bị công cụ:
- Đảm bảo máy tính đã cài đặt **Python 3.10+** và **Git**.
- Cài đặt **AWS CLI** (`aws`) trên máy tính của bạn và cấu hình tài khoản AWS (Access Key ID & Secret Access Key) bằng lệnh `aws configure`.
- Tạo một repository **Public mới, trống** trên GitHub.

### 2. Thiết lập môi trường dự án:
Mở terminal/command prompt và thực hiện lần lượt:

```bash
# Clone repo của bạn về (thay bằng URL repo của bạn)
git clone <URL_REPO_CUA_BAN> mlops-lab
cd mlops-lab

# Copy toàn bộ file hiện có của lab vào thư mục này (nếu bạn chưa copy)
# ...

# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows)
.venv\Scripts\activate
# (Nếu dùng Linux/Mac: source .venv/bin/activate)

# Cài đặt thư viện
pip install -r requirements.txt

# Tạo dữ liệu ban đầu
python generate_data.py
```
*Kết quả:* Sẽ có 3 file csv được tạo trong thư mục `data/`.

---

## Giai Đoạn 1: Thực Nghiệm Cục Bộ (Bước 1)

**Mục tiêu:** Xây dựng script huấn luyện model, log lại các tham số và metrics bằng MLflow.

### 1. Cấu hình MLflow:
Trong terminal, chạy:
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MLFLOW_ARTIFACT_ROOT=./mlartifacts
# Trên Windows cmd: set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

### 2. Tạo `params.yaml`:
Tạo file `params.yaml` ở thư mục gốc chứa các siêu tham số cho mô hình:
```yaml
n_estimators: 100
max_depth: 5
min_samples_split: 2
```

### 3. Hoàn thiện script `src/train.py`:
Mở file `src/train.py` (tạo nếu chưa có) và viết code theo template trong `tasks/buoc-1.md`.
**Các bước cần lập trình (điền TODO):**
- Dùng `pandas` để đọc `train_phase1.csv` và `eval.csv`.
- Tách `X` (features) và `y` (cột `target`).
- Dùng `with mlflow.start_run():` để theo dõi.
- Gọi `mlflow.log_params(params)`.
- Khởi tạo `RandomForestClassifier` với `**params` và fit nó với tập train.
- Dự đoán trên tập eval, tính `accuracy_score` và `f1_score`.
- Gọi `mlflow.log_metric()` cho accuracy và f1_score.
- Dùng `mlflow.sklearn.log_model()` lưu model.
- Lưu file `outputs/metrics.json` và `models/model.pkl`.

### 4. Chạy thực nghiệm:
- Chạy `python src/train.py`.
- Sửa `params.yaml` với các giá trị khác (VD: `n_estimators: 50, max_depth: 3`) và chạy lại `python src/train.py`. Lặp lại ít nhất 3 lần với 3 bộ tham số khác nhau.
- Chạy `mlflow ui --backend-store-uri sqlite:///mlflow.db`, vào `http://localhost:5000` để xem kết quả.
- Chọn bộ tham số tốt nhất (accuracy cao nhất) lưu vào `params.yaml`. Chụp màn hình MLflow UI.

---

## Giai Đoạn 2: Xây Dựng CI/CD Pipeline (Bước 2)

**Mục tiêu:** Push code lên GitHub -> Tự động test, train, eval và deploy lên máy chủ Cloud thông qua GitHub Actions và DVC.

### 1. Thiết lập Cloud Storage (AWS S3) & Credentials:
- Tạo một IAM User trên AWS với quyền truy cập Programmatic Access.
- Cấp quyền `AmazonS3FullAccess` cho user này (hoặc policy tùy chỉnh giới hạn trên bucket).
- Lưu lại **Access Key ID** và **Secret Access Key**.
- Chạy lệnh sau để tạo bucket S3 (thay bằng tên duy nhất toàn cầu):
  ```bash
  aws s3 mb s3://<BUCKET_NAME> --region us-east-1
  ```

### 2. Thiết lập DVC với AWS S3:
Vì máy tính đã chạy `aws configure`, DVC sẽ tự đọc credentials.
```bash
dvc init
dvc remote add -d myremote s3://<BUCKET_NAME>/dvc

dvc add data/train_phase1.csv data/eval.csv data/train_phase2.csv
git add data/*.dvc .gitignore .dvc/config
git commit -m "feat: track data with DVC"
dvc push
```

### 3. Thiết lập máy chủ triển khai (AWS EC2):
- Truy cập AWS Console > EC2 > Instances > Launch instances.
- Chọn Ubuntu 22.04 LTS, instance type `t2.micro`.
- Tạo một SSH Key Pair mới (`.pem`) và tải về máy tính để kết nối SSH.
- Ở mục Network settings, thiết lập Security Group: cho phép SSH (Port 22) từ mọi nơi, và Custom TCP (Port 8000) từ mọi nơi.
- Khởi chạy instance và lấy **Public IPv4 address**.
- SSH vào VM: `ssh -i path_to_key.pem ubuntu@<EC2_PUBLIC_IP>`
- Cài đặt môi trường trên EC2:
  ```bash
  sudo apt update && sudo apt install -y python3-pip
  pip3 install fastapi uvicorn scikit-learn joblib boto3
  mkdir -p ~/models ~/src
  ```

### 4. Viết Code API (`src/serve.py`):
- Khởi tạo app FastAPI.
- Code hàm tải model từ S3 (`boto3`):
  ```python
  import boto3
  s3 = boto3.client('s3')
  s3.download_file(GCS_BUCKET, GCS_MODEL_KEY, MODEL_PATH)
  ```
- Viết endpoint `GET /health` trả về `{"status": "ok"}`.
- Viết endpoint `POST /predict` nhận 12 features, chạy `model.predict`, trả về prediction và label text (thấp/trung_binh/cao).
- Copy `serve.py` lên VM: `scp -i path_to_key.pem src/serve.py ubuntu@<EC2_PUBLIC_IP>:~/src/serve.py`
- Trên VM, tạo systemd service `mlops-serve.service` để chạy ngầm (trong service cần thêm env `AWS_ACCESS_KEY_ID` và `AWS_SECRET_ACCESS_KEY`).

### 5. Cấu hình GitHub Secrets và SSH:
- Lên Repo GitHub > Settings > Secrets, thêm 5 secrets sau:
  1. `CLOUD_CREDENTIALS`: Định dạng JSON chứa credentials:
     `{"aws_access_key_id":"<KEY_CỦA_BẠN>","aws_secret_access_key":"<SECRET_CỦA_BẠN>"}`
  2. `CLOUD_BUCKET`: Tên bucket S3 của bạn.
  3. `VM_HOST`: IP Public của máy ảo EC2.
  4. `VM_USER`: Tên user SSH (`ubuntu`).
  5. `VM_SSH_KEY`: Nội dung file `.pem` hoặc private key SSH bạn tạo cho EC2.

### 6. Viết Test (`tests/test_train.py`):
- Viết các test case kiểm tra hàm `train()` sử dụng `tmp_path` trong pytest để tạo dữ liệu giả.
- Chạy `pytest tests/ -v` cục bộ để đảm bảo xanh hết.

### 7. Xây dựng Workflow CI/CD (`.github/workflows/mlops.yml`):
- Job **Test**: Chạy `pytest tests/ -v`.
- Job **Train**:
  - Dùng Secret `CLOUD_CREDENTIALS` parse ra để set ENV `AWS_ACCESS_KEY_ID` và `AWS_SECRET_ACCESS_KEY`.
  - Chạy `dvc pull`.
  - Chạy `python src/train.py`.
  - Đọc `accuracy` xuất ra biến output.
  - Upload `model.pkl` lên S3 bằng script python nhỏ với `boto3`.
- Job **Eval**: Chặn lại (exit 1) nếu accuracy nhận từ job Train < 0.70.
- Job **Deploy**: SSH vào VM, gọi `sudo systemctl restart mlops-serve`, sleep 5s, gọi `curl /health` để xác thực.

### 8. Đẩy code và kiểm tra CI/CD:
- Tạo `src/__init__.py` và `tests/__init__.py`.
- `git add .`, `git commit`, `git push origin main`.
- Quan sát tab Actions. Nếu cả 4 jobs xanh:
  - SSH vào VM start service: `sudo systemctl start mlops-serve`.
  - Test API bằng curl trên máy tính của bạn (tới địa chỉ public IP, port 8000). Chụp màn hình.

---

## Giai Đoạn 3: Mô Phỏng Huấn Luyện Liên Tục (Bước 3)

**Mục tiêu:** Thêm dữ liệu mới, DVC track sự thay đổi, đẩy lên Git và CI/CD tự động train lại model mới.

### 1. Bổ sung dữ liệu:
- Chạy `python add_new_data.py`. Dữ liệu sẽ tăng lên 5996 dòng.

### 2. Cập nhật DVC và kích hoạt CI/CD:
Thực hiện tuần tự:
```bash
# Báo cho DVC biết file csv thay đổi
dvc add data/train_phase1.csv

# Chỉ commit file .dvc lên Git (KHÔNG commit file csv)
git add data/train_phase1.csv.dvc
git commit -m "data: bổ sung 2998 mẫu dữ liệu mới (train_phase2)"

# BẮT BUỘC: Đẩy dữ liệu lên Cloud trước
dvc push

# Đẩy commit lên GitHub -> Sẽ kích hoạt workflow tự động chạy
git push origin main
```

### 3. Đánh giá:
- Lên GitHub Actions quan sát quy trình (Test -> Train -> Eval -> Deploy) hoàn toàn tự động.
- Mở file artifact `metrics.json` từ GitHub, so sánh `accuracy` và `f1_score` của Bước 2 so với Bước 3.
- Test lại endpoint predict với model mới cập nhật.

---
**Chúc bạn thực hiện bài Lab thành công! Hãy làm cẩn thận từng bước và đọc file log lỗi của GitHub Actions nếu có.**
