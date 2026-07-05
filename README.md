<p align="center">
  <img src="https://img.shields.io/badge/ISAS2026-Indoor%20Localization-blueviolet?style=for-the-badge" alt="ISAS2026"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Weighted%20F1-0.8351-success?style=for-the-badge" alt="F1 Score"/>
  <img src="https://img.shields.io/badge/Status-Final%20Submission-orange?style=for-the-badge" alt="Status"/>
</p>

---

# ISAS2026 — Xác Định Vị Trí (Phòng) Bằng BLE Trong Cơ Sở Chăm Sóc

Dự đoán phòng của caregiver trên tầng 5 từ tín hiệu RSSI của 25 BLE beacon.  
Dữ liệu được cung cấp bởi cuộc thi **ABC2026 Decode the Invisible Challenge**.

**Team:** Please Flob! I teach fish to swim

---

## Mục Lục

- [Bài Toán](#bài-toán)
- [Dữ Liệu](#dữ-liệu)
- [Pipeline — Tiến hóa qua 5 Version](#pipeline--tiến-hóa-qua-5-version)
- [Thuật Toán Sử Dụng](#thuật-toán-sử-dụng)
- [Kết Quả](#kết-quả)
- [Các Quyết Định Thiết Kế](#các-quyết-định-thiết-kế)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Cài Đặt](#cài-đặt)
- [Cách Chạy](#cách-chạy)

---

## Bài Toán

> Dựa trên tín hiệu BLE RSSI thu thập từ **25 beacon** được đặt tại tầng 5 của một cơ sở chăm sóc, dự đoán **phòng** mà caregiver đang ở.

**Mục tiêu:** 22 phòng (501–518, nurse station, cafeteria, kitchen, hallway, cleaning)

**Thách thức chính:** Mất cân bằng dữ liệu nghiêm trọng — phòng nurse station (9.361 mẫu) nhiều hơn **407 lần** so với phòng 516 (23 mẫu).

---

## Dữ Liệu

| Nguồn | Mô tả | Kích thước |
|-------|-------|------------|
| `Dataset/5f_label_loc_train.csv` | Nhãn huấn luyện (user_id=97, Location) | ~1.334 dòng |
| `Dataset/BLE Data/` | Dữ liệu BLE gốc (4107 file CSV) | ~1.67M dòng |
| `cleaned_labels.csv` | Nhãn sạch (chỉ Location, user_id=97) | 459 nhãn |
| `ble_train_4d.csv` | BLE pivot thành 25 beacon | 37.020 timestamps |
| `BLE_Test_predict.csv` | Dữ liệu test (user_id=90) | 62k dòng |

**Phân bố nhãn:**

| Phòng | Số mẫu (label) | Phòng | Số mẫu (label) |
|-------|----------------|-------|----------------|
| nurse station | 96 | 516 | 2 |
| cafeteria | 129 | 503 | 3 |
| kitchen | 77 | 510 | 4 |
| hallway | 41 | 518 | 4 |
| cleaning | 28 | 517 | 5 |
| 523 | 17 | ... | ... |

*Sau sliding window, số mẫu tăng lên do mỗi label bao phủ nhiều timestamp BLE. Tỉ lệ mất cân bằng sau window: nurse station ~9.361 vs 516 ~23.*

---

## Pipeline — Tiến hóa qua 5 Version

```
                           ble_train_4d.csv
                                │
                          Pivot theo timestamp
                                │
                          cleaned_labels.csv
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
           ┌─────────────────┐   ┌─────────────────┐
           │  V1 · V2 · V3  │   │  V3+4s · V4     │
           │  Buffer 0s     │   │  Buffer 4s      │
           └─────────┬───────┘   └────────┬────────┘
                     │                    │
                     ▼                    ▼
           ┌─────────────────┐   ┌─────────────────┐
           │  V1 · V2       │   │  V3 · V3+4s ·  │
           │  Không SG      │   │  V4            │
           │  (raw RSSI)    │   │  SG smoothing  │
           └─────────┬───────┘   └────────┬────────┘
                     │                    │
                     │    ┌───────────────┘
                     │    ▼
                     │   ┌─────────────────────┐
                     │   │  V4 (Relabel)       │
                     │   │  SMOTE-like: gán    │
                     │   │  nhãn từ majority   │
                     │   │  → minority         │
                     │   │  (thất bại)         │
                     │   └─────────────────────┘
                     │
                     └──────────┬───────────┘
                                ▼
                ┌─────────────────────────────┐
                │ Sliding window (k=5)        │
                │ mean, std, min, max × 25    │
                │ = 100 features             │
                └─────────────┬───────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  V1: 17-class  │ │  V2: 22-class  │ │  V3: 22-class  │
│  No weight     │ │  class_weight  │ │  class_weight  │
│  No SG         │ │  No SG         │ │  SG            │
│  W-F1=0.8315   │ │  W-F1=0.8246   │ │  W-F1=0.8351   │
│  M-F1=0.8193   │ │  M-F1=0.8191   │ │  M-F1=0.8336   │
└─────────────────┘ └───────┬─────────┘ └───────┬─────────┘
                            │                   │
                            ▼                   ▼
                   ┌─────────────────┐ ┌─────────────────┐
                   │  V3+4s: 22-cls │ │  V4: 22-class  │
                   │  class_weight  │ │  class_weight  │
                   │  SG + Buffer4s │ │  SG + Relabel  │
                   │  W-F1=0.8347   │ │  W-F1=0.8171 ❌│
                   │  M-F1=0.8350  │ │  M-F1=0.7713 ❌│
                   └─────────────────┘ └─────────────────┘
```

**Chi tiết kỹ thuật sliding window:**

| Kích thước window | k = 5 timestamps |
|-------------------|------------------|
| Đầu vào mỗi window | 5 × 25 giá trị RSSI |
| Đầu ra features | mean, std, min, max × 25 beacon = **100 features** |
| Mất mát padding | (k−1) = 4 mẫu đầu bị mất |

---

## Thuật Toán Sử Dụng

### 1. Savitzky-Golay Filter (Signal Smoothing)

**Mục đích:** Giảm nhiễu tần số cao trong tín hiệu RSSI thô do multipath fading, body shadowing, và orientation phone gây ra.

**Cách hoạt động:** Với mỗi beacon, trượt một cửa sổ gồm `window_length=5` điểm RSSI liên tiếp theo thời gian, fit một đa thức bậc `polyorder=2` bằng least squares, lấy giá trị smoothed tại tâm.

**Tại sao không phải là sinh dữ liệu ảo?**
- Không tạo mẫu mới, không tăng số lượng sample
- Chỉ thay đổi giá trị của sample đã tồn tại
- Giữ nguyên timestamp gốc
- Kết quả: W-F1 tăng +0.9%, Rare-F1 tăng +10.3%

### 2. Random Forest — Bộ phân loại chính

**Tại sao chọn Random Forest?**
- Mạnh mẽ với dữ liệu nhiều chiều (100 features)
- Không cần scaling / normalization
- Bắt được quan hệ phi tuyến giữa RSSI và vị trí
- Có sẵn feature importance để phân tích beacon nào quan trọng
- Xử lý đa cộng tuyến tốt (các beacon gần nhau có RSSI tương quan)

**Cấu hình:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=1
)
```

### 3. Stratified K-Fold Cross Validation

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 4. Sliding Window

Tổng hợp temporal trên k = 5 timestamp liên tiếp. Capture được pattern biến thiên RSSI mà một timestamp đơn lẻ không thể hiện.

### 5. Class Weighting

`balanced_subsample` gán trọng số tỉ lệ nghịch với tần suất class trong mỗi bootstrap sample:

| Phòng | Số mẫu | Trọng số | So với nurse station |
|-------|--------|----------|---------------------|
| 516 | ~23 | 47.2 | **407×** |
| nurse station | ~9.361 | 0.116 | 1× |

---

## Kết Quả

### So sánh tổng thể 5 Version

| Version | Thay đổi | W-F1 | M-F1 | Ghi chú |
|---|---|---|---|---|
| **V1** | 17-class, no weight, no SG | **0.8315** | 0.8193 | Gộp 6 phòng hiếm → other |
| **V2** | 22-class, +class_weight | 0.8246 | 0.8191 | Cân bằng, nhưng W-F1 giảm |
| **V3** | 22-class, +class_weight, +SG | **0.8351** 🏆 | **0.8336** | SG giúp tăng cả W & M |
| **V3+4s** | +Buffer 4s | 0.8347 | **0.8350** 🏆 | M-F1 cao nhất, phòng hiếm cải thiện |
| **V4** | +Relabeling ❌ | 0.8171 | 0.7713 | Thất bại, nhiễu dữ liệu |

### Per-class F1 cho 6 phòng hiếm

| Phòng | V1 | V2 | V3 | V3+4s | V4 ❌ |
|---|---|---|---|---|---|
| **503** | — | 0.875 | **0.875** | 0.860 | 0.672 |
| **505** | — | 0.844 | **0.938** 🚀 | 0.857 | 0.886 |
| **510** | — | 0.857 | **0.848** | 0.757 | 0.435 |
| **516** | — | 0.718 | 0.791 | **0.839** 🚀 | 0.653 |
| **517** | — | 0.739 | 0.800 | **0.844** 🚀 | 0.571 |
| **518** | — | 0.630 | 0.500 | **0.656** 🚀 | 0.300 |

### Hiệu suất mô hình cuối (V3)

| Chỉ số | Giá trị |
|--------|---------|
| Weighted F1 | **0.8351** |
| Macro F1 | **0.8336** |
| Số dự đoán (test) | 5.717 |
| SG filter | Savitzky-Golay (w=5, p=2) |
| Buffer | 0s (exact label boundaries) |

---

## Các Quyết Định Thiết Kế

| Quyết định | Lựa chọn | Lý do |
|-----------|---------|-------|
| Window size | k = 5 | Cân bằng temporal info vs số mẫu |
| Features | mean, std, min, max | 4 thống kê capture phân bố RSSI |
| SG filter | Savitzky-Golay (w=5, p=2) | Giảm nhiễu RSSI, cải thiện all metrics |
| Số cây | 100 | Cân bằng accuracy vs memory |
| Max depth | 20 | Chống overfit |
| Class weight | balanced_subsample | Xử lý mất cân bằng (22-class) |
| CV folds | 5 | Đánh giá đáng tin cậy |
| Buffer | 0s (V3) / 4s (V3+4s) | 4s cải thiện phòng hiếm nhẹ |
| SMOTE / Relabeling | ❌ Không dùng | Dữ liệu ảo / nhiễu không phản ánh không gian thật |
| Label data | Chỉ user_id = 97 | Labeler chính thức, bỏ user_id = 91 (tầng 2) |

---

## Cấu Trúc Dự Án

```
ISAS2026 Challenge/
├── Dataset/
│   ├── 5f_label_loc_train.csv      # Nhãn gốc
│   ├── BLE Data/                   # Raw BLE (4107 file, 333 MB)
│   └── 5th floor map.png           # Sơ đồ tầng 5
├── Bài báo trích dẫn/              # 4 reference papers
├── ble_train_4d.csv                # BLE pivot (183 MB, cần Git LFS)
├── cleaned_labels.csv              # Nhãn train đã làm sạch
├── BLE_Test_predict.csv            # Dữ liệu test
├── code/
│   ├── 01_model_selection.py       # V1: 17-class, chọn strategy
│   ├── 02_model_validation.py      # Báo cáo diagnostic V1
│   ├── 03_rare_room_test.py        # V2: 22-class + class_weight
│   ├── 04_final_training.py        # V3/V3+4s: 22-class + SG ± buffer
│   ├── 05_sampling_comparison.py   # SMOTE / RUS / CW so sánh
│   ├── 06_smoothing_comparison.py  # SG filter vs không SG
│   ├── 07_delay_analysis.py        # Phân tích delay cho adaptive buffer
│   └── 08_relabeling.py            # V4: relabeling (thất bại)
├── results/
│   ├── BLE_Test_predict_v1.csv     # 17-class
│   ├── BLE_Test_predict_v2.csv     # 22-class (V2)
│   ├── BLE_Test_predict_v3.csv     # 22-class + SG (FINAL)
│   └── BLE_Test_predict_v4_relabel.csv  # Relabel (thất bại)
├── notebooks/
│   ├── 1_Label location_train.ipynb
│   └── 2_BLE_train data_5f.ipynb
├── room_model.pkl                  # Model đã train
├── opencode.json                   # OpenCode + Ponytail plugin
└── README.md
```

---

## Clone & Push

```bash
git clone https://github.com/<user>/<repo>.git
cd "ISAS2026 Challenge"

git add .gitignore code/ results/ notebooks/ "Bài báo trích dẫn/" cleaned_labels.csv opencode.json README.md
git commit -m "ISAS2026 — BLE room prediction, 5 versions, W-F1 0.8351"
git remote add origin https://github.com/<user>/<repo>.git
git branch -M master
git push -u origin master
```

---

## Cài Đặt

```bash
# Yêu cầu Python 3.10+
pip install pandas numpy scikit-learn matplotlib seaborn scipy

# Tùy chọn: cho sampling comparison
pip install imbalanced-learn
```

---

## Cách Chạy

```bash
# V1: So sánh chiến lược (A vs D vs A+D)
python code/01_model_selection.py

# V2: 22-class + class_weight
python code/03_rare_room_test.py

# V3: 22-class + SG smoothing (FINAL)
python code/04_final_training.py

# V3+4s: thêm buffer 4s (sửa BUFFER trong code)
# code/04_final_training.py line 91: bu = pd.Timedelta(seconds=4)

# V4: Relabeling (experimental)
python code/08_relabeling.py
```

**Kết quả cuối:** `results/BLE_Test_predict_v3.csv` — 5.717 dự đoán phòng theo timestamp, W-F1 = 0.8351.

---

<p align="center">
  <b>Weighted F1: 0.8351 · Macro F1: 0.8336 · 22 phòng · SG smoothed</b>
</p>
