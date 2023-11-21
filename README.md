# Sentiment Analysis on Financial Documents 
เครื่องมือวิเคราะห์ทัศนคติและอารมณ์ในเอกสารทางการเงิน เป็นไลบรารี่สำหรับการวิเคราะห์ทัศนคติ (Aspect) และอารมณ์ (Sentiment) บนแบบฟอร์ม 56-1 ด้วยการเทรนจากข้อมูลแบบฟอร์ม 56-1 จำนวน 50 บริษัท ระหว่างปี 2558-2562 โดยสามารถรับไฟล์รูปแบบ pdf หรือข้อความ เป็นข้อมูลรับเข้า แล้วคืนค่าออกมาเป็นสัดส่วนทัศนคติ และอารมณ์ที่มีต่อแต่ละส่วนของเอกสาร เช่น ส่วนการบริหารจัดการความเสี่ยง ส่วนการวิเคราะห์และคำอธิบายของฝ่ายจัดการ และส่วนการขับเคลื่อนธุรกิจเพื่อความยั่งยืน

การพัฒนาเครื่องมือนี้ได้รับการสนับสนุนจาก กองทุนส่งเสริมการพัฒนาตลาดทุน (Capital Market Development Fund: CMDF)

## Installation

```
!pip install -r requirement.txt
```

## Prediction

```
from thai_report_analyzer import reportAnalyzer
```

กรณีให้แบบฟอร์ม 56-1 ในรูปแบบไฟล์ pdf เป็นข้อมูลนำเข้า

```
anlyzer = reportAnalyzer("content/path_to_file.pdf")
sentiment = analyzer.predict_sentiment(0:100)
aspect = analyzer.predict_aspect(0:100)
```

กรณีให้ข้อความเป็นข้อมูลนำเข้า

```
anlyzer = reportAnalyzer()
text = "บริษัทล้มละลายจึงถูกควบรวมกิจการในไตรมาสที่ 2"
sentiment = analyzer.predict_sentiment(text)
aspect = analyzer.predict_aspect(text)
```

## Dataset
ชุดข้อมูลนี้ได้เก็บรวบรวมมาจากเอกสารแบบฟอร์ม 56-1 จากบริษัทรวม 50 แห่ง ตั้งแต่ปี พ.ศ. 2558 - 2562 และสร้างป้ายกำกับข้อมูล (data annotation) ในระดับกลุ่มของประโยคว่าสะท้อนอารมณ์และทัศนคติ จำนวน 12,258 กลุ่มประโยค ตามประเภทต่อไปนี้ 

| ประเภททัศนคติ (Aspects) |                                                                                                                                                                                       คำนิยาม/คำอธิบาย                                                                                                                                                                                      |
|:---------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Brand                 | ภาพลักษณ์บริษัท/ตราสินค้า (แบรนด์) - แบรนด์ รวมถึง ภาพลักษณ์ตราสินค้า ภาพลักษณ์องค์กร ภาพลักษณ์สินค้า - การตลาด รวมถึง การโฆษณา PR รางวัลการตลาด การส่งเสริมการขาย การทำสปอนเซอร์                                                                                                                                                                                                                                  |
| Product/Service       | ผลิตภัณฑ์หรือบริการของบริษัท - การประกาศ ออกสินค้า/บริการใหม่ การทดลองตลาดสินค้า/บริการ - การเปลี่ยนแปลง อัพเกรด/ดาวน์เกรด เรียกคืน อนุมัติ - ความร่วมมือกับบริษัทอื่น เช่น การทำ licensing พันธมิตร (alliance) หุ้นส่วนธุรกิจ (partnership) การทำ MOU, Joint Venture                                                                                                                                                           |
| Environment           | การดำเนินงานด้านสิ่งแวดล้อม - นโยบายด้านสิ่งแวดล้อม นิเวศวิทยา ภาวะโลกร้อน (Global Warming) การเปลี่ยนแปลงสภาพภูมิอากาศ (Climate Change) - การสร้างของเสีย การปล่อยมลพิษ - กิจกรรม CSR (Corporate Social Responsibility) ต่าง ๆ ทั้งในด้านสิ่งแวดล้อม                                                                                                                                                                 |
| Social & People       | สังคม และผู้คน - การจ้างงาน - การเปลี่ยนแปลงในการจ้างงานของพนักงานบริษัท (การจ้าง หรือ เลิกจ้าง) - ค่าใช้จ่ายในการจ้างงาน (compensation) - การเปลี่ยนแปลงผู้บริหาร (เช่น CEO, ผู้บริหารระดับสูง) - การหยุดงาน (strike) - ประเด็นอื่น ๆ ที่เกี่ยวกับการจ้างงาน  กิจกรรม CSR ต่าง ๆ ทั้งในด้านสังคม ที่เกี่ยวกับพนักงาน แรงงาน ลูกค้า ชุมชน ท้องถิ่น หรือผู้มีส่วนได้ส่วนเสียอื่น ๆ                                                                            |
| Governance            | ธรรมาภิบาลของบริษัท - การเปลี่ยนแปลงของคณะกรรมการบริษัท (Board of Directors) - นโยบายการกำกับดูแลบริษัท และบริษัทย่อย - ความโปร่งใสในการดำเนินงาน จริยธรรม การตรวจสอบของผู้บริหาร                                                                                                                                                                                                                             |
| Economics             | การบรรยายถึงเศรษฐกิจมหภาค ที่อาจส่งผลต่อบริษัท - สภาวะเศรษฐกิจของประเทศและโลก นโยบายเศรษฐกิจต่าง ๆ นโยบายการค้าขายระหว่างประเทศ เช่น FTA (Free trade agreements) -ดัชนีทางเศรษฐกิจต่าง ๆ เช่น GDP อัตราดอกเบี้ย อัตราเงินเฟ้อ อัตราการว่างงาน รายได้ประชาชาติ อัตราแลกเปลี่ยนค่าเงิน - แนวโน้มเศรษฐกิจใน อุตสาหกรรม ประเทศ และโลก                                                                                              |
| Political             | การเมือง - การเปลี่ยนแปลงทางการเมือง เช่น การเลือกตั้ง การทำรัฐประหาร การเคลื่อนไหวทางการเมือง ความไม่สงบทางการเมือง สงคราม - นโยบายของภาครัฐ - นโยบายภาษี                                                                                                                                                                                                                                                |
| Legal                 | ข้อพิพาททางกฎหมาย หรือการตัดสินใจที่เกี่ยวข้องกับกฎหมาย รวมถึง การสอบสวน การกล่าวหา การฟ้องร้อง คดีความ การถูกดำเนินคดี การฉ้อโกง การฟอกเงิน การยอมความ การจ่ายค่าเสียหาย คำพิพากษา กฎหมาย และประเด็นทางกฎหมายอื่นๆ                                                                                                                                                                                                   |
| Dividend              | เงินปันผล คือ เงินจ่ายให้แก่ผู้ถือหุ้นของบริษัท - การจ่ายเงินปันผล อาจมาในรูปของ เงินสด หุ้น หรือสินทรัพย์รูปแบบอื่น - สังเกตการเปลี่ยนแปลงที่เกี่ยวกับเงินปันผลในด้าน การคาดการณ์ (forecast) การรายงาน การประกาศจ่าย                                                                                                                                                                                                              |
| Investment            | การลงทุน - เงินลงทุน (capital expenditure) ในตัวบริษัท บริษัทย่อยหรือร่วม สาขา การลงทุนในโครงสร้างการผลิต (เช่น โรงงาน) การลงทุนในสินค้าหรือบริการ - การลงทุนในการวิจัยและพัฒนา (Research & Development) - เหตุการณ์ที่เกี่ยวกับ โรงงาน ตึกสำนักงาน อาคาร ร้านค้า สาขา โกดัง หรืออสังหาริมทรัพย์อื่น ๆ - ยกเว้น การควบรวมกิจการ (M&A)                                                                                                  |
| M&A                   | การควบรวมกิจการของบริษัท (Merger and Acquisition) - Merger คือ การที่บริษัทตั้งแต่ 2 บริษัทขึ้นไปทำการควบรวมกิจการเข้าด้วยกันแล้วเกิดเป็นบริษัทใหม่ - Acquisition คือ การที่บริษัทหนึ่ง เข้าไปซื้อกิจการบางส่วนหรือทั้งหมด ของอีกบริษัทหนึ่ง ซึ่งเราสามารถแบ่งออกได้เป็น 2 กรณีด้วยกัน  * Share Acquisition คือ การที่ผู้ซื้อเข้ามาซื้อหุ้นของบริษัทบางส่วน หรือทั้งหมด  * Asset / Business Acquisition คือ การที่ผู้ซื้อเข้ามาซื้อทรัพย์สิน, หน่วยธุรกิจบางส่วนหรือทั้งหมด ของกิจการ |
| Profit/Loss           | ผลประกอบการบริษัท - นับรวมไปถึง รายได้ (Revenue) ยอดขาย (Sales) ต้นทุนขาย (Costs of Goods Sold) ค่าใช้จ่ายต่าง ๆ (Expenses) - ตัวเลขทางการเงิน (Financials) หรืออัตราส่วนทางการเงิน (Financial Ratios) ต่าง ๆ - กำไร (หรือ ขาดทุน) สุทธิ คือ รายได้หลังหักค่าใช้จ่ายทั้งหมด - กำไรสุทธิ = รายได้ - ต้นทุนขาย - ค่าใช้จ่ายในการขายและบริหาร - ค่าใช้จ่ายดอกเบี้ย - ภาษี - รวมถึงการเปลี่ยนแปลงราคาหุ้นในตลาดหลักทรัพย์                           |
| Rating                | อันดับความน่าเชื่อถือของบริษัท - การจัดเรตติ้ง การจัดอันดับความน่าเชื่อถือของตัวองค์กร หรือ การจัดอันดับความน่าเชื่อถือของตราสารหนี้แต่ละตัว ที่จะสะท้อนความสามารถในการชำระหนี้ของผู้ออกตราสาร - สถาบันจัดอันดับความน่าเชื่อถือในไทยมี 2 แห่งคือ บจก.ทริสเรทติ้ง (TRIS) และ บจก. ฟิทช์ เรทติ้งส์ (Fitch) - ข้อเสนอแนะ หรือ คำแนะนำของนักวิเคราะห์ (เช่น คำแนะนำ ซื้อ/ขาย/ถือ) เกี่ยวกับ รวมถึงการเปลี่ยนแปลงคำแนะนำด้วย                                             |
| Financing             | การกู้ยืมเงิน (loan) การทำ syndicated loan การออกหุ้นกู้ (bond) การเพิ่มทุนในตลาดหลักทรัพย์ การซื้อหุ้นกลับคืน (stock repurchase) การให้กู้ยืมระหว่างกันบริษัทที่เกียวข้อง/บริษัทลูก การทำ IPO (initial public offering) การทำ private placement หุ้น การทำ tender offer การเพิ่ม/ลดทุน จาก VC, angel investor                                                                                                                    |
| Technology            | การเปลี่ยนแปลงด้านเทคโนโลยี สารสนเทศ การใช้ automation การใช้ AI นวัตกรรมต่าง ๆ การเข้าถึง (Access), licensing, patent และ ทรัพย์สินทางปัญญาเทคโนโลยี                                                                                                                                                                                                                                                    |
| Others                | หัวข้ออื่น ๆ หัวข้อการเปลี่ยนแปลงในด้านอื่น ๆ นอกเหนือจากที่กล่าวข้างต้น อาทิ เช่น การเปลี่ยนแปลงเทคโนโลยี ภัยพิบัติ โรคระบาด                                                                                                                                                                                                                                                                                       |

| ประเภทขั้วอารมณ์ (Sentiment) |                                                                                                                            คำนิยาม/คำอธิบาย                                                                                                                           |
|:-------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Negative                  | ความรู้สึกที่เป็นลบ โดยยึดโยงจากมุมมองนักลงทุนทั่วไป หรือนักวิเคราะห์หลักทรัพย์/การเงิน - ข้อความสื่อถึงการเปลี่ยนแปลง หรือ ผลกระทบที่เป็น แง่ลบ ต่อประเภทของทัศนคติ (Aspect)  - ข้อความที่เป็น ลบ เป็นได้ทั้ง ข้อเท็จจริง ความเห็น ความคิด อารมณ์ หรือ การตัดสินใจ เกี่ยวกับบริษัท                                         |
| Neutral                   | ความรู้สึกที่เป็นกลาง โดยยึดโยงจากมุมมองนักลงทุนทั่วไป หรือนักวิเคราะห์หลักทรัพย์/การเงิน - ข้อความ ไม่ได้สื่อ ถึงการเปลี่ยนแปลง หรือ ผลกระทบที่เป็นด้านใดด้านหนึ่ง ต่อประเภทของทัศนคติ - ข้อความที่อาจจะเป็น กลาง เป็นเพียงคำกล่าวของข้อเท็จจริงเกี่ยวกับบริษัท - ความเห็น ความคิด อารมณ์ ที่แสดงนั้นเป็นไปอย่างกลาง ๆ (moderate) |
| Positive                  | ความรู้สึกที่เป็นบวก โดยยึดโยงจากมุมมองนักลงทุนทั่วไป หรือนักวิเคราะห์หลักทรัพย์/การเงิน - ข้อความสื่อถึงการเปลี่ยนแปลง หรือ ผลกระทบที่เป็น แง่บวก ต่อประเภทของทัศนคติ (Aspect) - ข้อความที่เป็น บวก เป็นได้ทั้ง ข้อเท็จจริง ความเห็น ความคิด อารมณ์ หรือ การตัดสินใจ เกี่ยวกับบริษัท                                       |

### Annotation
ข้อมูลเหล่านี้ถูกกำกับโดยผู้มีพื้นฐานทางด้านเศรษฐศาสตร์ จำนวน 4 คน โดยก่อนการกำกับข้อมูลจริง นักกำกับข้อมูลเหล่านี้ได้ทดสอบกำกับข้อมูลจนสามารถกำกับข้อมูลได้อย่างตรงกัน ด้วยคะแนน Cohen kappa เฉลี่ยระหว่างกลุ่มเท่ากับ 0.73 (Substantial) ในการกำกับข้อมูลทัศนคติ และได้คะแนนเท่ากับ 0.77 (Substantial) ในการกำกับข้อมูลอารมณ์

อ้างอิง Level of Agreement จาก Landis & Koch (1977)

## Training

ในการพัฒนาแบบจำลองคณะวิจัยได้แบ่งชุดข้อมูลออกเป็น 3 ชุด ได้แก่ 
- (70%) Training set จำนวน 8,191 กลุ่มประโยค ระหว่างปี พ.ศ. 2558 -  2561 ของบางบริษัท  
- (15%) Validation set จำนวน 1,756 กลุ่มประโยค  ปี พ.ศ. 2562
- (15%) Test set จำนวน 1,755 กลุ่มประโยค ระหว่างปี พ.ศ. 2561 - 2562 ของบางบริษัท

**สัดส่วนป้ายกำกับข้อมูล**

|      ทัศนคติ      |  Training set  | Validation set |   Test set   |
|:---------------:|:--------------:|:--------------:|:------------:|
| Brand           |   113 (1.38%)  |   47 (2.68%)   |  29 (1.65%)  |
| Product/Service |   894 (10.9%)  |  243 (13.84%)  |  159 (9.06%) |
| Environment     |   491 (5.99%)  |   150 (8.54%)  |  96 (5.47%)  |
| Social & People | 1,688 (20.61%) |   346 (19.7%)  | 318 (18.12%) |
| Governance      |  943 (11.51%)  |   209 (11.9%)  | 202 (11.51%) |
| Economics       |   676 (8.25%)  |   116 (6.61%)  |  135 (7.69%) |
| Political       |   103 (1.26%)  |   21 (1.20%)   |  24 (1.37%)  |
| Legal           |   143 (1.75%)  |   31 (1.77%)   |  28 (1.60%)  |
| Dividend        |   64 (0.78%)   |    7 (0.45%)   |  13 (0.74%)  |
| Investment      |   283 (3.46%)  |   81 (4.61%)   |  81 (4.62%)  |
| M&A             |   37 (0.45%)   |   19 (1.08%)   |  13 (0.74%)  |
| Profit/Loss     | 1,777 (21.69%) |  349 (19.87%)  | 380 (21.65%) |
| Rating          |    8 (0.10%)   |     0 (0%)     |   3 (0.17%)  |
| Financing       |   221 (2.70%)  |   39 (2.22%)   |  38 (2.17%)  |
| Technology      |   81 (0.99%)   |   23 (1.31%)   |  29 (1.65%)  |
| Others          |   669 (8.17%)  |   75 (4.27%)   | 207 (11.79%) |

|  ขั้วอารมณ์ |  Training set  | Validation set |   Test set   |
|:--------:|:--------------:|:--------------:|:------------:|
| Negative | 1,316 (16.07%) |  226 (15.15%)  | 323 (18.40%) |
| Neutral  | 3,751 (45.79%) |  686 (39.07%)  | 821 (46.78%) |
| Positive | 3,124 (38.14%) |  804 (45.79%)  | 611 (34.81%) |

**ไฮเปอร์พารามิเตอร์**
- ทัศนคติ (Aspect)
    - Learning Rate = 3e-5
    - Batch Size = 16
    - Epoch = 5
    - Weight Decay = 0.01
- ขั้วอารมณ์ (Sentiment)
    - Learning Rate = 5e-5
    - Batch Size = 16
    - Epoch = 5
    - Weight Decay = 0.01

สามารถดูรายละเอียดการเทรนได้ที่ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I9jESy1WBgtrv9bU_Rnr_rMfJg6be0xH?usp=sharing)

## Performance
เครื่องมือถูกพัฒนาโดยการพัฒนาต่อจากแบบจำลอง ```airesearch/wangchanberta-base-att-spm-uncased``` บนชุดข้อมูลและไฮเปอร์พารามิเตอร์ที่ได้กล่าวไป และได้ผลการทดลอง ดังนี้

**ทัศนคติ (Aspect)**

|     Metrics   | Score     |
|:---------------|:---------:|
|     Accuracy    | 0.79 |
|    Micro avg - Precision | 0.79 |
|    Micro avg - Recall    | 0.79 |
|    Micro avg - F1        | 0.79 |
|    Macro avg - Precision | 0.72 |
|    Macro avg - Recall    | 0.65 |
|    Macro avg - F1        | 0.66 |
|   Weighted avg - Precision | 0.79 |
|   Weighted avg - Recall    | 0.79 |
|   Weighted avg - F1        | 0.78 |
|   Brand - Precision |  0.8 |
|   Brand - Recall    |  0.8 |
|   Brand - F1        | 0.67 |
|   Dividend - Precision | 0.81 |
|   Dividend - Recall    |   1  |
|   Dividend - F1        |  0.9 |
|   Economics - Precision | 0.79 |
|   Economics - Recall    | 0.88 |
|   Economics - F1        | 0.79 |
|   Environment - Precision | 0.84 |
|   Environment - Recall    | 0.88 |
|   Environment - F1        | 0.86 |
|    Financing - Precision | 0.49 |
|    Financing - Recall    | 0.68 |
|    Financing - F1        | 0.57 |
|    Governance - Precision | 0.72 |
|    Governance - Recall    | 0.88 |
|    Governance - F1        | 0.79 |
|    Investment  - Precision | 0.79 |
|    Investment - Recall    | 0.51 |
|    Investment - F1        | 0.62 |
|      Legal - Precision | 0.62 |
|      Legal - Recall    | 0.71 |
|      Legal - F1        | 0.67 |
|       M&A - Precision |   1  |
|       M&A - Recall    | 0.23 |
|       M&A - F1        | 0.38 |
|      Others - Precision | 0.84 |
|      Others - Recall    | 0.62 |
|      Others - F1        | 0.71 |
|    Political - Precision | 0.68 |
|    Political - Recall    | 0.54 |
|    Political - F1        |  0.6 |
| Product/Service - Precision | 0.67 |
| Product/Service - Recall    | 0.67 |
| Product/Service - F1        | 0.84 |
|   Profit/Loss - Precision | 0.84 |
|   Profit/Loss - Recall    | 0.91 |
|   Profit/Loss - F1        | 0.87 |
|      Rating - Precision |   0  |
|   Rating - Recall    |   0  |
|   Rating - F1        |   0  |
|  Social&People - Precision | 0.85 |
|  Social&People - Recall    | 0.86 |
|  Social&People - F1        | 0.86 |
|    Technology - Precision |  0.8 |
|  Technology - Recall    | 0.69 |
|  Technology - F1        | 0.74 |

**ขั้วอารมณ์ (Sentiment)**

|:    Aspects :|  Metrics      |
|:------------|:---------:|
|   Accuracy | 0.77 |
|   Micro avg  | Precision | 0.77 |
|   Micro avg - Recall    | 0.77 |
|   Micro avg - F1        | 0.77 |
|   Macro avg - Precision | 0.78 |
|   Macro avg - Recall    | 0.75 |
|   Macro avg - F1        | 0.76 |
| Weighted avg - Precision | 0.77 |
| Weighted avg - Recall    | 0.77 |
| Weighted avg - F1        | 0.77 |
|   Negative - Precision | 0.80 |
|   Negative - Recall    | 0.69 |
|   Negative - F1        | 0.74 |
|    Neutral - Precision | 0.76 |
|    Neutral - Recall    | 0.80 |
|    Neutral - F1        | 0.78 |
|   Positive - Precision | 0.77 |
|   Positive - Recall    | 0.78 |
|   Positive - F1        | 0.77 |

## License
[Creative Commons Attribution 4.0 International Public License (CC-by)](https://creativecommons.org/licenses/by/4.0/)

## Citations
หากคุณใช้ ``` thai_report_analyzer ``` ในโปรเจคหรืองานวิจัยของคุณ กรุณาอ้างอิงไลบรารี่ ตามรูปแบบดังนี้


