# 🛒 Shopping Cart Association Analysis

Phân tích dữ liệu bán lẻ để tìm ra mối quan hệ giữa các sản phẩm thường được mua cùng nhau bằng các kỹ thuật **Association Rule Mining** (Apriori / FP-Growth). Project triển khai pipeline đầy đủ từ xử lý dữ liệu → phân tích → khai thác luật → sinh báo cáo.

---

## ✨ Features

- Làm sạch dữ liệu & xử lý giá trị lỗi
- Xây dựng basket matrix (transaction × product)
- Khai phá tập mục phổ biến (Frequent itemsets)
- Sinh luật kết hợp (Association Rules)
- Các chỉ số:
  - Support
  - Confidence
  - Lift
- Visualization với:
  - bar chart
  - scatter plot
  - network graph
  - interactive Plotly
- Tự động hóa pipeline bằng **Papermill**

---

## 📂 Project Structure

shopping_cart_analysis/
├── data/
│ ├── raw/
│ │ └── online_retail.csv
│ └── processed/
│ ├── cleaned_uk_data.csv
│ ├── basket_bool.parquet
│ └── rules_apriori_filtered.csv
├── notebooks/
│ ├── 01_preprocessing_and_eda.ipynb
│ ├── 02_basket_preparation.ipynb
│ ├── 03_apriori_modeling.ipynb
│ └── runs/
├── src/
│ └── shopping_cart_library.py
├── run_papermill.py
├── requirements.txt
└── README.md

yaml
Sao chép mã

---

## 🚀 Installation

```bash
git clone <your_repo_url>
cd shopping_cart_analysis
pip install -r requirements.txt
🧪 Data Preparation
Đặt file gốc vào:

bash
Sao chép mã
data/raw/online_retail.csv
File output sẽ được sinh tự động vào:

bash
Sao chép mã
data/processed/
▶️ Run Pipeline (Recommended)
Chạy toàn bộ phân tích chỉ với 1 lệnh:

bash
Sao chép mã
python run_papermill.py
Kết quả sinh ra:

bash
Sao chép mã
data/processed/cleaned_uk_data.csv
data/processed/basket_bool.parquet
data/processed/rules_apriori_filtered.csv
notebooks/runs/03_apriori_modeling_run.ipynb
🔧 Changing Parameters
Các tham số có thể chỉnh trong run_papermill.py:

python
Sao chép mã
MIN_SUPPORT=0.01
MAX_LEN=3
FILTER_MIN_CONF=0.3
FILTER_MIN_LIFT=1.2
Hoặc sửa trong cell PARAMETERS của mỗi notebook để chạy với cấu hình khác nhau.

📊 Visualization & Results
Notebook 03 hiển thị các biểu đồ sau:

Top luật theo Lift

Top luật theo Confidence

Scatter Support–Confidence–Lift

Network Graph giữa các sản phẩm

Biểu đồ Plotly tương tác

Bạn có thể export sang HTML:

bash
Sao chép mã
jupyter nbconvert notebooks/runs/03_apriori_modeling_run.ipynb --to html
🧠 Ứng dụng thực tế
Product recommendation

Cross-selling strategy

Combo gợi ý sản phẩm

Phân tích hành vi mua hàng

Sắp xếp sản phẩm tại siêu thị

🛠 Tech Stack
Công nghệ	Mục đích
Python	language
Pandas	data processing
MLxtend	Apriori / FP-Growth
Papermill	batch notebook run
Matplotlib / Seaborn	visualization
Plotly	interactive visualization
Jupyter	notebook engine

📈 Roadmap
 Thêm FP-Growth notebook (04)

 Streamlit dashboard để lọc luật

 Xuất báo cáo PDF tự động

 Kết hợp luật với RFM segmentation

👤 Author
Project thực hiện bởi:
Your Name / Trangle / Nguyễn ??? (bạn điền vào nhé 😉)

📄 License
MIT — sử dụng tự do cho nghiên cứu, học thuật và ứng dụng nội bộ.