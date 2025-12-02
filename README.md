# Merl (មើល)

Merl (មើល) is a lightweight bilingual (Khmer/English) OCR that pairs my own recognizer with Surya OCR for layout detection, tables, latex ocr, and reading order.
Why not use Surya alone? Its built-in recognizer still struggles with the wide variety of Khmer fonts, so I wasn’t satisfied with the accuracy. I don’t have a public paper for this achitecture — everything comes from thousands of experiments across different model architectures and datasets.


## Installation
```bash
# install from PyPI
pip install merl

# install from source
git clone https://github.com/MetythornPenn/merl.git && cd merl
pip install -e .
```

## Getting started
```python
from IPython.display import display
from merl import (
    Merl,
    draw_document_boxes,
    gather_bboxes,
    document_to_markdown,
)

image_path = "samples/sample_1.png"

ocr = Merl()
ocr.load()     

# 1) Line-level OCR
# text = ocr.recognize_line(image_path)
# print("Line text:", text)

# 2) Document-level OCR (Recommend)
doc = ocr.analyze_document(image_path)
print("Device:", doc.device)
print("Reading order:", doc.reading_order)
print("Lines:", [line.text for line in doc.lines])
print("Timings:", doc.timings)

# 3) LaTeX recognition 
# latex = ocr.recognize_latex("path/to/formula.png")

# 4) Visualize bounding boxes 
annotated = draw_document_boxes(image_path, doc, show_layout=True, show_detections=True)
display(annotated)  # in notebooks; annotated.save("annotated.jpg") to persist

# 5) Export to Markdown
md = document_to_markdown(doc)
print("Markdown preview:\n", md[:500])

```


## Example output

## Example output

**Sample 1**
<p align="center">
  <img src="samples/sample_1.png" alt="Sample 1 Input" width="48%" />
  <img src="samples/sample_1_annotated.png" alt="Sample 1 Annotated" width="48%" />
</p>
<pre>
គណនេយ្យ តាមការណែនាំរបស់អ្នកផ្ទះ ដែលតាមពិតគាត់មាន
សមត្ថភាពក្នុងផ្នែកអាយធីច្រើនជាង។ ក្រោយមក គាត់ក៏ប្តូរមក
ផ្នែកអាយធីវិញ ព្រោះទៅមិនរួចក្នុងគណនេយ្យ។ គាត់ត្រូវខាត
ពេលរៀនមុខវិជ្ជាសងសាលាវិញ ហើយថែមទាំងត្រូវរៀនមូលដ្ឋាន
គ្រឹះឡើងវិញទៀតផង។
មនុស្សយើងអាចមានជំនាញច្រើនក្នុងខ្លួន។ តែបើយល់ល្អ ការមាន
ជំនាញមិនលើសពីលី គឺប្រសើរបំផុត។ អ្នកខ្លះមានជំនាញផ្នែកទី
ផ្សារផង និងជំនាញផ្នែកកុំព្យូទ័រទូទៅដែរ។ សូមកុំច្រលំ ការមាន
ជំនាញច្រើន និងការរៀនច្រើនសាលាឱ្យសោះ។ រៀនច្រើនសាលា
មិនប្រាកដថាធ្វើឱ្យយើងមានជំនាញច្បាស់លាស់នោះទេ។ "ជំនាញ
មិនស្ថិតលើសាលាទេ តែស្ថិតនៅលើការខំច្រឹងរបស់យើង។ រឿងពីរា
ខុសគ្នា ហើយផលវាក៏ខុសគ្មាដែរ។ យើងអាចបង្កើនជំនាញមួយឱ្យ
ឆ្អិនល្មមសិន ចាំចាប់ផ្តើមជំនាញថ្មីមួយទៀត។
នេះជារឿងដែលខ្ញុំស្តាយជាងគេបំផុត។ ខ្ញុំរៀនដោយមិនបានចាប់
យកជំនាញពីដំបូង ហេតុនេះធ្វើឱ្យយឹតពេលនៅពេលចេញធ្វើការ។
ពេលធ្វើការ ក្រៅពីត្រូវរវល់នឹងការងារហើយ ខ្ញុំត្រូវពង្រឹងជំនាញ
របស់ខ្ញុំក្នុងពេលតែមួយទៀតផង។ បើចង់ចំណេញពេល និងមិន
ពិបាកខ្លួន គួរតែអភិវឌ្ឍជំនាញតាំងពីដើមឱ្យហើយ។
</pre>

**Sample 2**
<p align="center">
  <img src="samples/sample_2.png" alt="Sample 2 Input" width="48%" />
  <img src="samples/sample_2_annotated.png" alt="Sample 2 Annotated" width="48%" />
</p>
<pre>
ផ្នោតលើជំនាញ
យើងនឹងលឺគ្រូគ្រប់មុខវិជ្ជា ចូលមកពន្យល់យើងថា មុខវិជ្ជាដែល
គាត់បង្រៀនសំខាន់ប៉ុណ្ណោះ។ គ្រូណាក៏និយាយបែបនោះដែរ ព្រោះ
បើវាមិនសំខាន់ សាលាក៏មិនជូលគាត់មកបង្រៀនដែរ។ ប៉ុន្តែ យើង
ត្រវិសួរថា តើវាសំខាន់សម្រាប់យើងដែរទេ?
នៅពេលយើងជ្រើសរើសថារៀនអ្វីរួចហើយ យើងត្រវតែចាប់
ជំនាញមួយឱ្យបាន។ កុំខ្វល់ពីរឿងផ្សេងទៀត ធ្វើយ៉ាងណាក៏ដោយ
ត្រវតែចេះអ្វីមួយឱ្យច្បាស់។ ការរៀនត្រឹមតែមួយដឹង មិនអាចជួយ
អ្វីយើងបានទេ។
ពេលវេលាដែលល្អបំផុតក្នុងការរៀនចាប់យកជំទាញ គឺឆ្នាំទីមួយ។
យើងត្រូវសម្រុកវៀនអ្វីដែលសំខាន់ សម្រាប់ជំនាញដែលយើងរើស
តាំងពីបើកឆាកទៅឱ្យហើយ បើមិនអញ្ចឹងទេ ប្រាកដជាយឺតពេល។
កាលដែលខ្ញុំរៀនឆ្នាំទីមួយ ខ្ញុំមិនបានដឹងរឿងនេះទេ។ ខ្ញុំរវល់តែ
ខ្វល់ជាមួយនឹងមុខវិជ្ជាដែលគ្រូប្រាប់ថាសំខាន់ ហើយភ្លេចថាខ្លួនឯង
ក៏មានមុខវិជ្ជាសំខាន់ដែរ។ មានពេលខ្លះ ខ្ញុំចំណាយពេលពីរសប្តាហ៍
</pre>

**Sample 3**
<p align="center">
  <img src="samples/sample_3.png" alt="Sample 3 Input" width="48%" />
  <img src="samples/sample_3_annotated.png" alt="Sample 3 Annotated" width="48%" />
</p>
<pre>ទំនាក់ទំនងផ្សាយពាណិជ្ជកម្ម</pre>

**Sample 4**
<p align="center">
  <img src="samples/sample_4.png" alt="Sample 4 Input" width="48%" />
  <img src="samples/sample_4_annotated.png" alt="Sample 4 Annotated" width="48%" />
</p>
<pre>
លោក ស សុខា បានបញ្ជាក់ថា ដើម្បីដោះស្រាយបញ្ហានេះ គឺផ្តើមចេញពីក្រសួងមហាផ្ទៃផ្ទាល់តែម្តង។ លោកចានណែនាំអង្គ
ភាពជំនាញ ឱ្យរៀបចំយន្តការក្រុមការងារសហការស្រាវជ្រាវបង្ក្រាបបទល្មើសចោកប្រាស់ដូចបានបញ្ជាក់ខាងលើនេះ។
តាមរយៈយន្តការនេះ លោកបានអះអាងថា មុន្ត្រីជំនាញប្រចាំការ នឹងធ្វើការងារនេះ២៤ម៉ោង ជារៀងរាល់ថ្ងៃ ដើម្បីសហការ
ជាមួយតំណាងធនាគារនិងគ្រឹះស្ថានមីក្រូហិរញ្ញវត្ថុ ក្នុងករណីទទួលបានបណ្តឹងពីជនរងគ្រោះ ក្នុងនោះក៏រួមទាំងសហការ
ស្មើសុំបង្កកប្រាក់ក្មុងគណនីជនល្មើសជាដើម។
</pre>

**Sample 5**
<p align="center">
  <img src="samples/sample_5.png" alt="Sample 5 Input" width="48%" />
  <img src="samples/sample_5_annotated.png" alt="Sample 5 Annotated" width="48%" />
</p>
<pre>បើតាមលោករដ្ឋមន្ត្រីមហាផ្ទៃ រូបភាពនៃការបន្លំបោកប្រាស់ទាំងនោះ មានដូចជា ការបន្លំខ្លួនធ្វើជាមន្រ្តីនេះគាតារា</pre>
