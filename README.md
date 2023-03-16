# Data Analysis of Video Game Sales
## 1. Import Data & Python Packages
เริ่มต้น เราได้ import library ที่จำเป็นในการวิเคราห์ข้อมูลซึ่งประกอบด้วย
  1. NumPy
  2. Pandas
  3. Matplotlib
  4. Seaborn
  
```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
plt.rc("font", size=14) #dictionary objects
import seaborn as sns
import matplotlib.ticker as mtick
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)
```
และได้ upload Data: Global Video Game Sales ที่ได้รับมาจาก Kaggle โดย Data ชุดนี้มีข้อมูลในแต่ละ column คือ

  1. Rank = อันดับของเกมอ้างอิงตาม Global_Sales
  2. Name = ชื่อเกม
  3. Platform = แพลตฟอร์มของเกม
  4. Year = ปีที่วางจำหน่าย
  5. Genre = ประเภทเกม
  6. Publisher = ผู้จัดจำหน่าย
  7. NA_Sales = ยอดขายในอเมริกาเหนือ
  8. EU_Sales = ยอดขายในยุโรป
  9. JP_Sales = ยอดขายในญี่ปุ่น
  10. Other_Sales = ยอดขายในภูมิภาคอื่น ๆ
  11. Global_Sales = ยอดขายรวมทั่วโลก
และมีจำนวนข้อมูลทั้งหมด 16,598 row
```python
# Read CSV data file into DataFrame
df = pd.read_csv("vgsales.csv")

# preview data
df.head()
```
