[README.md](https://github.com/user-attachments/files/29622876/README.md)
<p align="center">
  <img src="https://img.shields.io/badge/ABC2026-Decode%20the%20Invisible-blueviolet?style=for-the-badge" alt="ABC2026"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Weighted%20F1-0.8244-success?style=for-the-badge" alt="F1 Score"/>
  <img src="https://img.shields.io/badge/Status-Final%20Submission-orange?style=for-the-badge" alt="Status"/>
</p>

---

# 🏥 ISAS2026 — Xác Định Vị Trí (Phòng) Bằng BLE Trong Cơ Sở Chăm Sóc

**Decode the Invisible Challenge** — Dự đoán phòng của caregiver trên tầng 5 từ tín hiệu RSSI của 25 BLE beacon.

**Team:** Please Flob! I teach fish to swim

---

## 📋 Mục Lục

- [Bài Toán](#-bài-toán)
- [Dữ Liệu](#-dữ-liệu)
- [Pipeline Tổng Quan](#-pipeline-tổng-quan)
- [Thuật Toán Sử Dụng](#-thuật-toán-sử-dụng)
- [Kết Quả](#-kết-quả)
- [Các Quyết Định Thiết Kế](#-các-quyết-định-thiết-kế)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Cài Đặt](#-cài-đặt)
- [Cách Chạy](#-cách-chạy)
- [Ghi Chú Dữ Liệu](#-ghi-chú-dữ-liệu)
- [Lời Cảm Ơn](#-lời-cảm-ơn)

---

## 🎯 Bài Toán

> Dựa trên tín hiệu BLE RSSI thu thập từ **25 beacon** được đặt tại tầng 5 của một cơ sở chăm sóc, dự đoán **phòng** mà caregiver đang ở.

**Mục tiêu:** 22 phòng (501–518, nurse_station, cafeteria, living, dining)

**Thách thức chính:** Mất cân bằng dữ liệu nghiêm trọng — một số phòng (ví dụ nurse_station 9.361 mẫu) nhiều hơn **400 lần** so với phòng hiếm (ví dụ phòng 516 chỉ 23 mẫu).

---

## 📊 Dữ Liệu

| Nguồn | Mô tả | Kích thước |
|-------|-------|------------|
| `Dataset/5f_label_loc_train.csv` | Nhãn huấn luyện (user_id=97, Location + Activity) | ~1.334 dòng |
| `Dataset/BLE Data/` | Dữ liệu BLE gốc (4107 file CSV) | ~1.67M dòng |
| `cleaned_labels.csv` | Nhãn sạch (chỉ Location) | 459 nhãn |
| `ble_train_4d.csv` | BLE pivot thành 25 beacon | 37.020 timestamps |
| `BLE_Test_predict.csv` | Dữ liệu test (23 beacon) | 62k dòng |

**Phân bố nhãn nổi bật:**

| Phòng | Số mẫu | Phòng | Số mẫu |
|-------|--------|-------|--------|
| nurse_station | 9.361 | 516 | 23 |
| living | 6.082 | 503 | 32 |
| cafeteria | 3.506 | 510 | 37 |
| 518 | 42 | 505 | 44 |
| 517 | 42 | ... | ... |

---

## 🔄 Pipeline Tổng Quan

```
Dataset/ ───┬── 5f_label_loc_train.csv ──→ cleaned_labels.csv
             └── BLE Data/ (4107 files) ──→ ble_train_4d.csv
                                                     │
                                                     ▼
                               ┌─────────────────────────────┐
                               │   TIỀN XỬ LÝ DỮ LIỆU        │
                               │  • Pivot theo timestamp      │
                               │    → 37.020 timestamps       │
                               │  • Merge với nhãn            │
                               │    → 23.905 mẫu có nhãn      │
                               │  • Sliding window (k=5)      │
                               │    → 100 features / 23.901   │
                               └──────────┬──────────────────┘
                                          │
                                          ▼
              ┌─────────────────────────────────────────────────┐
              │        01_model_selection.py                    │
              │  So sánh 3 strategy (A / D / A+D)              │
              │  → D (no_weight) tốt nhất: W-F1=0.8315 (17-class)│
              └──────────────────────┬──────────────────────────┘
                                     │
                                     ▼
              ┌─────────────────────────────────────────────────┐
              │        02_model_validation.py                   │
              │  Báo cáo per-class, confusion matrix,           │
              │  kiểm tra temporal consistency                  │
              │  → Độ chính xác phòng: 93.19%                   │
              └──────────────────────┬──────────────────────────┘
                                     │
                                     ▼
              ┌─────────────────────────────────────────────────┐
              │        03_rare_room_test.py                     │
              │  Mô hình 22-class có đoán được 6 phòng hiếm?   │
              │  → CÓ! F1=0.630–0.875 cho phòng hiếm            │
              │  → Quyết định: dùng 22-class (không gộp)        │
              └──────────────────────┬──────────────────────────┘
                                     │
                                     ▼
              ┌─────────────────────────────────────────────────┐
              │        04_final_training.py                     │
              │  • RF 22-class, class_weight='balanced'        │
              │  • 100 cây, max_depth=20                       │
              │  • Predict test → BLE_Test_predict_v5.csv      │
              └─────────────────────────────────────────────────┘
```

**Chi tiết kỹ thuật sliding window:**

| Kích thước window | k = 5 timestamps |
|-------------------|------------------|
| Đầu vào mỗi window | 5 × 25 giá trị RSSI |
| Đầu ra features | mean, std, min, max × 25 beacon = **100 features** |
| Mất mát padding | (k−1) = 4 mẫu đầu bị mất |

---

## 🧠 Thuật Toán Sử Dụng

### 1. Random Forest — Bộ phân loại chính

**Tại sao chọn Random Forest?**
- Mạnh mẽ với dữ liệu nhiều chiều (100 features)
- Không cần scaling/normalization
- Bắt được quan hệ phi tuyến giữa RSSI và vị trí
- Có sẵn feature importance để phân tích beacon nào quan trọng
- Xử lý đa cộng tuyến tốt (các beacon gần nhau có RSSI tương quan)

**Cấu hình:**
```python
RandomForestClassifier(
    n_estimators=100,                    # 100 cây
    max_depth=20,                        # Chống overfit
    class_weight='balanced_subsample',   # Ưu tiên phòng hiếm
    random_state=42,
    n_jobs=1                             # Tránh lỗi memory paging
)
```

### 2. Stratified K-Fold Cross Validation

Chia 5 folds, giữ nguyên tỉ lệ class trong mỗi fold:
```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 3. Sliding Window

Tổng hợp temporal trên k=5 timestamp liên tiếp. Capture được pattern biến thiên RSSI mà một timestamp đơn lẻ không thể hiện.

### 4. Class Weighting (Cân bằng trọng số)

`balanced_subsample` gán trọng số tỉ lệ nghịch với tần suất class trong mỗi bootstrap sample:

| Phòng | Số mẫu | Trọng số | So với nurse_station |
|-------|--------|----------|---------------------|
| 516 | 23 | 47.2 | **407×** |
| nurse_station | 9.361 | 0.116 | 1× |

### 5. Label Encoding

`LabelEncoder` chuyển tên phòng thành số nguyên 0–21 để sklearn có thể xử lý.

---

## 📈 Kết Quả

### So sánh chiến lược

| Strategy | Weighted F1 | Macro F1 | Classes | Ghi chú |
|----------|-------------|----------|---------|---------|
| **A** (class_weight) | 0.8137 | 0.8063 | 22 | Dự đoán được tất cả, F1 thấp hơn |
| **D** (no_weight) | **0.8315** | 0.8193 | 17 | W-F1 cao nhất nhưng gộp 6 phòng hiếm |
| **A+D** (class_weight + thresh=0.3) | 0.8050 | — | 17 | Threshold làm tụt F1 |
| **✅ V5 (A, 22-class)** | **0.8244** | **0.8158** | **22** | **Chọn cuối cùng** |

### Hiệu suất mô hình cuối (V5)

| Chỉ số | Giá trị |
|--------|---------|
| Weighted F1 | **0.8244** |
| Macro F1 | **0.8158** |
| Độ chính xác số phòng | **93.19%** (1983/2128) |
| Confidence trung bình (test) | 0.613 |

### Hiệu suất trên 6 phòng hiếm (<50 mẫu)

| Phòng | Số mẫu | F1 Score |
|-------|--------|----------|
| 503 | 32 | **0.875** |
| 505 | 44 | **0.844** |
| 510 | 37 | **0.857** |
| 516 | **23** | **0.718** |
| 517 | 42 | **0.739** |
| 518 | 42 | **0.630** |

---

## 🔑 Các Quyết Định Thiết Kế

| Quyết định | Lựa chọn | Lý do |
|-----------|---------|-------|
| Window size | k=5 | Cân bằng temporal info vs số mẫu |
| Features | mean, std, min, max | 4 thống kê capture phân bố RSSI |
| Số cây | 100 | Cân bằng accuracy vs memory |
| Max depth | 20 | Chống overfit |
| Class weight | balanced_subsample | Xử lý mất cân bằng (22-class) |
| CV folds | 5 | Đánh giá đáng tin cậy |
| SMOTE | ❌ Không dùng | Dữ liệu ảo không phản ánh không gian thật |

---

## 📁 Cấu Trúc Dự Án

```
ABC2026 Sozolab Challenge/
├── Dataset/                        # Dữ liệu gốc (đã zip)
├── ble_train_4d.csv                # BLE per-packet pivot (183 MB)
├── cleaned_labels.csv              # Nhãn train đã làm sạch
├── BLE_Test_predict.csv            # Dữ liệu test
├── code/
│   ├── 01_model_selection.py       # Chọn strategy → V4
│   ├── 02_model_validation.py      # Báo cáo diagnostic
│   ├── 03_rare_room_test.py        # Kiểm tra phòng hiếm
│   └── 04_final_training.py        # Mô hình cuối → V5
├── results/
│   ├── BLE_Test_predict_v4.csv     # Dự phòng 17-class
│   └── BLE_Test_predict_v5.csv     # ✅ Kết quả CUỐI 22-class
├── notebooks/
│   ├── 1_Label location_train.ipynb
│   └── 2_BLE_train data_5f.ipynb
├── room_model.pkl                  # Model đã train
└── README.md                       # File này
```

---

## ⚙️ Cài Đặt

```bash
# Yêu cầu Python 3.10+
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🚀 Cách Chạy

Chạy lần lượt từng script:

```bash
# Bước 1: So sánh chiến lược (A vs D vs A+D)
python code/01_model_selection.py

# Bước 2: Kiểm tra chất lượng mô hình
python code/02_model_validation.py

# Bước 3: Kiểm tra khả năng dự đoán phòng hiếm
python code/03_rare_room_test.py

# Bước 4: Train mô hình cuối và dự đoán test
python code/04_final_training.py
```

**Kết quả:** `results/BLE_Test_predict_v5.csv` — 5.717 dự đoán phòng theo timestamp.

---

## 🧪 Ghi Chú Dữ Liệu

- **Dữ liệu train** chỉ dùng `user_id=97` (người gắn nhãn do ban tổ chức cung cấp)
- **Làm sạch**: loại bỏ timestamp trùng, bỏ activity không phải Location, gộp khoảng thời gian chồng lấn
- **Sliding window padding**: 4 timestamp đầu mỗi session bị bỏ (không đủ window)
- **Nhận dạng activity** đang chờ dữ liệu accelerometer từ ban tổ chức

---

## 🏆 Lời Cảm Ơn

- Ban tổ chức **ABC2026 Decode the Invisible Challenge** đã cung cấp dữ liệu
- Các thành viên team **Sozolab**
- Xây dựng với ❤️ bằng scikit-learn, pandas và numpy

---

<p align="center">
  <b>Weighted F1: 0.8244 · Macro F1: 0.8158 · 22 phòng được dự đoán</b>
</p>
