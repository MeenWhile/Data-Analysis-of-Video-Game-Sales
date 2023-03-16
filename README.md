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

![image](https://user-images.githubusercontent.com/125643589/225664370-b618d4e4-0511-4527-838c-4526a1c1ef5b.png)
![image](https://user-images.githubusercontent.com/125643589/225664736-8c5ea210-a4f0-4164-80fb-33d2548b7429.png)

## 2. Data Quality & Missing Value Assessment
ต่อมา เราได้ทำการตรวจสอบ data ว่ามีข้อมูลไหนขาดหายไปหรือไม่
โดยผลลัพธ์ที่ได้คือเราได้รับรู้ว่า
  1. ข้อมูลใน column Year หายไป 271 ข้อมูล
  2. ข้อมูลใน column Publisher หายไป 58 ข้อมูล

![image](https://user-images.githubusercontent.com/125643589/225665729-ef2a5d26-19f5-4d65-80c3-50084f086a72.png)

### 2.1. Year - Missing Values
เราได้ทำการเจาะลึกลงไปยัง column Year และได้รับรู้ว่า จำนวนข้อมูลที่หายไป มีอัตราส่วน 1.63% เมื่อเทียบกับข้อมูลทั้ง column ซึ่งเรามองว่า อัตราส่วนข้อมูลที่หายไปนี้ไม่ได้มีจำนวนที่เยอะเท่าไหร่ จึงไม่จำเป็นที่จะต้องตัด column นี้ออกไป

นอกจากนั้น เราได้ทำการหาค่า mean และ median ของ column นี้ เพื่อเตรียมพร้อมที่จะ fill ข้อมูลที่ขาดหายไป โดยผลลัพธ์ที่ได้คือ
  1. ค่า mean ของ column Year คือ 2006.41
  2. ค่า median ของ column Year คือ 2007
  
![image](https://user-images.githubusercontent.com/125643589/225666210-29117475-94b8-4031-8ee5-8a0918cc05b1.png)

### 2.2. Publisher - Missing Values
จากนั้น เราได้ทำการเจาะลึกลงไปยัง column Publisher ต่อ โดยได้รับรู้ว่า จำนวนข้อมูลที่ขาดหายไปของ column นี้ มีอัตราส่วน 0.35% เมื่อเทียบกับข้อมูลทั้ง column ซึ่งถือได้ว่าจำนวนข้อมูลที่ขาดหายไปมีจำนวนที่น้อย จึงไม่จำเป็นต้องลบ column นี้ทิ้งเช่นกัน

และเมื่อได้รู้อัตราส่วนของข้อมูลที่หายไปแล้ว ต่อมา เราได้ดูลักษณะของข้อมูลใน column นี้ เพื่อเตรียมพร้อมที่จะ fill ข้อมูลที่ขาดหายไปได้อย่างเหมาะสม
โดยเราได้พบว่า "Electronic Arts" มีจำนวนมากที่สุดใน column นี้ ซึ่งมีจำนวนทั้งหมด 1351 ข้อมูล

![image](https://user-images.githubusercontent.com/125643589/225666469-554b8138-c57a-4829-8289-66012cf5ac03.png)

### 2.3. Final Adjustments to Data
หลังจากที่ได้เจาะลึกไปยังแต่ละ column ว่าข้อมูลที่ขาดหายไปมีลักษณะโดยรวมเป็นยังไงแล้ว ทีนี้เราก็เริ่มต้น fill ข้อมูลที่ขาดหายไป โดย
  1. column Year -> เราเลือกใช้ค่า median ซึ่งก็คือ 2007 ในการ fill ข้อมูล เนื่องจากค่า median นี้สามารถช่วยกรองค่า outlier ได้ รวมถึง value ที่ได้จะเป็นเลขจำนวนเต็ม ซึ่งเป็น value ที่เหมาะสมสำหรับ column Year
  2. column Publisher -> สำหรับ column นี้ ตอนแรกเราตั้งใจที่จะ fill ข้อมูลด้วยค่า "Electronic Arts" ซึ่งเป็นข้อมูลที่มีจำนวนเยอะที่สุดใน column นี้ แต่ว่า เมื่อลองมองถึงลักษณะของ Publisher หรือก็คือ "ผู้จัดทำเกม" แล้ว เราคิดว่าการที่ข้อมูลของ Publisher หายไป มีความเป็นไปได้สูงว่าอาจจะเป็น "ผู้จัดทำเกมรายย่อย" ที่ผู้รวบรวม data ไม่สามารถหาข้อมูลในส่วนได้ เราจึงเลือกที่จะ fill ข้อมูลของ column นี้ด้วยค่า "empty value" แทน

โดยเมื่อเราได้ fill ข้อมูลทั้งหมดแล้ว เราก็ได้เช็คข้อมูลซ้ำอีกครั้งว่ายังเหลือข้อมูลที่ขาดหายไปอยู่หรือไม่
ซึ่งผลลัพธ์ก็คือ จำนวน missing value ในตอนนี้เป็น 0

![image](https://user-images.githubusercontent.com/125643589/225666738-d847abea-8c2a-4f07-ba24-2d6de8cf6516.png)

จากนั้น เราได้ทำการเปรียบเทียบข้อมูลของ column Year กับ column Publisher ก่อนและหลัง fill data เพื่อตรวจสอบว่าข้อมูลก่อนและหลังการแก้ไข มีการเปลี่ยนแปลงมากจนเกินไปหรือไม่ ซึ่งจากผลลัพธ์ที่ได้นั้น เรามองว่าการเปลี่ยนแปลงข้อมูลใน column ทั้งสองนี้ อยู่ในระดับที่เหมาะสม

![image](https://user-images.githubusercontent.com/125643589/225666885-a51d1a3c-7431-43c9-88c4-755920b83df8.png)
![image](https://user-images.githubusercontent.com/125643589/225666964-550f964d-83ff-45f2-a99c-ca124be2f6c3.png)
