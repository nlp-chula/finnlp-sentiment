# Sentiment Analysis on Financial Documents 


## Requirements

## Prediction

กรณีให้ไฟล์รายงานรูปแบบ pdf เป็นข้อมูลนำเข้า

```
anlyzer = report.Analyzer("content/path_to_pdf.pdf")
sentiment = analyzer.predict_sentiment(0:100)
aspect = analyzer.predict_aspect(0:100)
```
กรณีให้ข้อความเป็นข้อมูลนำเข้า

```
anlyzer = report.Analyzer()
text = "บริษัทล้มละลายจึงถูกควบรวมกิจการในไตรมาสที่ 2"
sentiment = analyzer.predict_sentiment(text)
aspect = analyzer.predict_aspect(text)
```


## Dataset
จำนวน train/test/dev

## Annotation
โดยใคร เกณฑ์คืออะไร ค่า Kappa score

### Training

สัดส่วนป้ายกำกับข้อมูล

|a|b|c|d|
|:---:|:---:|:---:|:---:|
|a|b|c|d|

ไฮเปอร์พารามิเตอร์


สามารถดูรายละเอียดการเทรนได้ที่ [Colab]()

## Performance

aspect

|a|b|c|d|
|:---:|:---:|:---:|:---:|
|a|b|c|d|

sentiment

|a|b|c|d|
|:---:|:---:|:---:|:---:|
|a|b|c|d|
