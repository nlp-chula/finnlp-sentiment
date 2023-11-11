# clean PDF text
from pythainlp.corpus import thai_words
from pythainlp import word_tokenize
from pythainlp.util import normalize
from pypdf import PdfReader
import pythainlp

import kenlm
# ocr pdf
import pytesseract
import fitz
from PIL import Image

import re, os
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import (
    load_dataset, load_metric,
    Dataset,
    DatasetDict,
    Features, Sequence, ClassLabel, Value
)





class reportAnalyzer:

    correct_dict = {'':'่',
                '>':'้',
                'ํ':'',
                "":"่",
                "":"้",
                "":"ี",
                "":"์",
                "":"็",
                "":"ั",
                "":"ื",
                "":"้",
                "":"๋",
                "":"ิ",
                "":"๊",
                "":"์",
                "":"่",
                "":"ึ",
                "":"้",
                'ํา':'ำ',
                '':'',
                '':'',
                '':'',
                ' ':'',
                '–':'-'}
    
    def __init__(self, pdf_file_path:str=None, file_type:str=[None,'pdf','ocr']):
        if pdf_file_path is not None:
            self.pdf_file_path = pdf_file_path
            self.kenlm_model = kenlm.Model('kenlm/sixgram.arpa')
            if file_type == 'pdf':
                line_list = self.read_pdf()
            elif file_type == 'ocr':
                line_list = self.read_pdf_ocr()
            section_dict = self.extract_section(line_list)
            self.extract_df = self.build_dataset(section_dict)
        self.sentiment_prob = None
        self.aspect_prob = None

    def calc_spelling_score(self, text):
        """
        Calculates the spelling score of the given text using the KenLM language model.

        Args:
            text (str): The text to calculate the spelling score for.

        Returns:
            float: The spelling score of the given text.
        """
        log_score = 0.0
        for i, (logprob, length, oov) in enumerate(self.kenlm_model.full_scores(text)):
            log_score += logprob
        return log_score
    
    def clean_text(self, text):
        """
        Cleans the given text by replacing errors with their correct counterparts and fixing common Thai spelling mistakes.
        
        Args:
        - text (str): The text to be cleaned.
        
        Returns:
        - str: The cleaned text.
        """        
        text_list = []
        index_list = []
        tone_list = ['','่','้','๊','๋']
        
        for err, corr in self.correct_dict.items():
            text = text.replace(err, corr)
        
        for char in text: # list comprehension
            text_list.append(char)
            if char == 'า' or char == 'ำ':
                index = len(text_list)-1
                index_list.append(index)
            elif char in ['ั','ึ','ิ','ี','ื','ำ']:
                index = len(text_list)-1
                index_list.append(index)
                text_list.append('')
            
            for index in index_list:
                if text_list[index] == 'า' or text_list[index] == 'ำ':
                    text_list[index] = 'า'
                    word_ah = '\r'.join(text_list)
                    text_list[index] = 'ำ'
                    word_um = '\r'.join(text_list)
                    if self.calc_spelling_score(word_ah) > self.calc_spelling_score(word_um):
                        text_list[index] = 'า'
                    else:
                        text_list[index] = 'ำ'
                else:
                    if len(text_list) < index+1:
                        pass
                    else:
                        best = -100000
                        for tone in tone_list:
                            text_list[index+1] = tone
                            word = '\r'.join(text_list)
                            point = self.calc_spelling_score(word)
                            if point > best:
                                best = point
                                best_tone = tone
                        text_list[index+1] = best_tone
                text = ''.join(text_list)
        return text

    def read_pdf_ocr(self):
        """
        Reads in a PDF file and performs OCR (Optical Character Recognition) on it to extract text.
        The function first reads in the PDF file and extracts text from each page. Then, it converts each page to a JPG image
        and performs OCR on the image to extract text. The extracted text is cleaned up and returned as a list of strings.
        
        Returns:
        - extracted_text: A list of strings, where each string represents a line of text extracted from the PDF.
        """
        
        # Step 1: Read in the PDF and extract text
        pdf_file = self.pdf_file_path
        pdf_document = fitz.open(pdf_file)
        output_folder = f"{pdf_file.split('/')[-1].split('.')[0]}"
        os.makedirs(output_folder, exist_ok=True)

        # Step 2: Convert pages to JPG and perform OCR
        text_images = []

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            image = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_path = os.path.join(output_folder, f"page_{page_num}.jpg")
            image.save(img_path, "JPEG")
            text_images.append(img_path)

        # Step 3: Perform OCR on the JPG images
        extracted_text = []
        for img_path in text_images:
            text = pytesseract.image_to_string(Image.open(img_path), lang='tha')
            text_list = [self.clean_text(line) for line in text.split('\n')]
            extracted_text.extend(text_list)

        # Cleanup: Delete temporary JPG images if desired
        for img_path in text_images:
            os.remove(img_path)

        # Close the PDF document
        pdf_document.close()

        return extracted_text

    def read_pdf(self):
        """
        Reads a PDF file and returns a list of cleaned text from each page.

        Returns:
        page_list (list): A list of cleaned text from each page of the PDF file.
        """
        page_list = []
        reader = PdfReader(self.pdf_file_path)
        for num_page, page in enumerate(reader.pages):
            text = page.extract_text()
            line_list = text.split('\n')
            for line_idx, line in enumerate(line_list):
                line_list[line_idx] = self.clean_text(line)
            line_list.pop()
            page_list += line_list # one-page
        return page_list
    
    def extract_section(self, page_list):
        """
        Extracts a specific section from a list of text lines based on the PDF file path.

        Args:
            page_list (list): A list of text lines.

        Returns:
            dict: A dictionary containing the extracted section as a string value, with the section type as the key.
        """
        
        result_dict = {}
        for line_idx, line in enumerate(page_list):
            if line == '' or re.match('^\s$',line): # clean blank space
                page_list.pop(line_idx)
        index_start = None
        index_end = None
        if 'BUSINESS' in self.pdf_file_path.upper(): # หาปัจจัยความเสี่ยง
            key = 'risk'
            keyword_start = '3.ปัจจัยความเสี่ยง'
            alternative = '3.ปัจจัยเสี่ยง|3.ปัจจัยเสี่ยง'
            keyword_end = '4.ทรัพย์สินที่ใช้ในการประกอบธุรกิจ'
        elif 'MANAGEMENT' in self.pdf_file_path.upper():
            key = 'sustainability'
            keyword_start = '10.ความรับผิดชอบต่อสังคม'
            alternative = '10.การพัฒนาอย่างยั่งยืน|10.การดำเนินธุรกิจอย่างยั่งยืน'
            keyword_end = 'การควบคุมภายในและการบริหารจัดการความเสี่ยง'
        elif 'FINANCIAL' in self.pdf_file_path.upper():
            key = 'md&a'
            keyword_start = '14.การวิเคราะห์และคำอธิบาย'
            alternative = '14.คำอธิบายและบทวิเคราะห์|14.คำอธิบายและการวิเคราะห์'
            keyword_end = 'การรับรองความถูกต้องของข้อมูล'
        else:
            for i in range(1,4):
                if i == 1:
                    key = 'risk'
                    keyword_start = 'ปัจจัยความเสี่ยง'
                    alternative = 'ปัจจัยเสี่ยง|ปัจจัยเสี่ยง'
                    keyword_end = 'ทรัพย์สินที่ใช้ในการประกอบธุรกิจ'
                elif i == 2:
                    key = 'sustainability'
                    keyword_start = 'ความรับผิดชอบต่อสังคม'
                    alternative = 'การพัฒนาอย่างยั่งยืน|การดำเนินธุรกิจอย่างยั่งยืน'
                    keyword_end = 'การควบคุมภายในและการบริหารจัดการความเสี่ยง'
                elif i == 3:
                    key = 'md&a'
                    keyword_start = 'การวิเคราะห์และคำอธิบายของฝ่ายจัดการ'
                    alternative = 'คำอธิบายและบทวิเคราะห์ของฝ่ายบริหาร|คำอธิบายและการวิเคราะห์ของฝ่ายจัดการ'
                    keyword_end = 'การรับรองความถูกต้องของข้อมูล'
                
                for line_idx,line in enumerate(page_list):
                    if line == '' or re.match('^\s$',line): # clean blank space
                        page_list.pop(line_idx)
                for line_idx,line in enumerate(page_list):
                    clean = re.sub('\s+', ' ', line)
                    page_list[line_idx] = clean
                    pattern = keyword_start + '|' + alternative
                    if re.search(pattern, clean) and index_start is None: #the first one
                        index_start = line_idx
                    elif keyword_end in clean:
                        index_end = line_idx
                if index_end is None:
                    index_end = len(page_list)-1
                result_dict[key] = '\n'.join(page_list[index_start:index_end])
            return result_dict
        
        for line_idx,line in enumerate(page_list):
            if line == '' or re.match('^\s$',line): # clean blank space
                page_list.pop(line_idx)
        
        for line_idx, line in enumerate(page_list):
            clean = re.sub('\s+', ' ', line)
            page_list[line_idx] = clean
            pattern = keyword_start + '|' + alternative
            if re.search(pattern, clean) and index_start is None: #the first one
                index_start = line_idx
            elif keyword_end in clean:
                index_end = line_idx

        if index_end is None:
            index_end = len(page_list)-1
        result_dict[key] = '\n'.join(page_list[index_start:index_end])
        return result_dict
    
    def likely_table(self, text, threshold):
        """
        Determines whether a given text is likely to be a row in a table based on the number of numeric characters it contains.

        Args:
            text (str): The text to analyze.
            threshold (float): The percentage of numeric characters required for a row to be considered "possible".

        Returns:
            int: 1 if the text is a possible row in a table, 0 otherwise.
        """
        
        line_len = len(text)
        num_cnt = 0
        for c in text:
            if num_cnt > (line_len*(threshold/100)):
                return 1 # 'possible rows'
            elif c.isnumeric():
                num_cnt += 1
        return 0 #'not a row in table'
    
    def incomplete_sentence(self, text):
        """
        Determines whether a given text is a complete or incomplete sentence based on its length and whether it starts with a number.

        Args:
            text (str): The text to be analyzed.

        Returns:
            int: 1 if the text is an incomplete sentence, 0 if it is a complete sentence.
        """
        
        if re.search('^\d{1,2}[\.\-]?\d{0,2}', text): # if the text starts with a number
            if len(text) > 57:
                return 0 # complete sentence
            else:
                return 1 # incomplete sentence
        elif len(text) > 57: # 57 is the average length of an incomplete sentence in annotation pilot
            return 0 # complete sentence
        else:
            return 1 # incomplete sentence
    
    def build_dataset(self, result_dict):
        """
        Builds a pandas DataFrame from the given `result_dict` dictionary, which contains the parsed report data.
        The resulting DataFrame contains columns for the firm name, year, section, and paragraph text. It also adds two
        additional columns: `isTable` and `isIncomplete`, which are binary indicators of whether the paragraph is likely
        a table or an incomplete sentence, respectively. Paragraphs that are identified as tables or incomplete sentences
        are filtered out of the final DataFrame.

        Args:
            result_dict (dict): A dictionary containing the parsed report data, organized by firm, year, section, and paragraph.

        Returns:
            pandas.DataFrame: A DataFrame containing the firm name, year, section, and paragraph text for each paragraph
            in the report data that is not identified as a table or incomplete sentence.
        """        
        firm_list = []
        year_list = []
        section_list = []
        para_list = []
        for firm in result_dict.keys():
            for year in result_dict[firm].keys():
                for section in result_dict[firm][year].keys():
                    for para in result_dict[firm][year][section]:
                        firm_list.append(firm)
                        year_list.append(year)
                        section_list.append(section)
                        para_list.append(para)
        df = pd.DataFrame({'firm': firm_list, 'year': year_list, 'section': section_list, 'paragraph': para_list})
        df['isTable'] = df['paragraph'].apply(lambda x: self.likely_table(str(x), threshold=20))
        df['isIncomplete'] = df['paragraph'].apply(lambda x: self.incomplete_sentence(str(x)))

        df = df[(df['isTable'] == 0) & (df['isIncomplete'] == 0)].reset_index()[['firm', 'year', 'section', 'paragraph']]
        return df

    def predict_sentiment(self, start:int, end:int, text:str=None):
        """
        Predicts the sentiment of a given text or a range of texts from the `extract_df` DataFrame.
        
        Args:
        - start (int): The starting index of the range of texts to predict sentiment for.
        - end (int): The ending index of the range of texts to predict sentiment for.
        - text (str, optional): The text to predict sentiment for. If None, the method will use the texts from the `extract_df` DataFrame.
        
        Returns:
        - A dictionary containing the sentiment ratios of the predicted texts. The keys are the sentiment labels ('Positive', 'Negative', 'Neutral') and the values are the ratios of each sentiment label in the predicted texts.
        """
             
        model_url = 'nlp-chula/augment-sentiment-finnlp-th'
        tokenizer = AutoTokenizer.from_pretrained(model_url, max_length = 412)
        tokenizer_kwargs = {'truncation':True,'max_length':412}
        sentiment_classifier = pipeline(task='text-classification',
                                            tokenizer=tokenizer,
                                            model = model_url, 
                                            truncation =True)
        if text is None:
            self.sentiment_prob = sentiment_classifier(self.extract_df['paragraph'][start:end], **tokenizer_kwargs)
        else:
            self.sentiment_prob = sentiment_classifier(text, **tokenizer_kwargs)
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

        # Counting occurrences of each sentiment type
        for data in self.sentiment_prob:
            sentiment_counts[data['label']] += 1

        # Calculating sentiment ratios
        total_samples = len(self.sentiment_prob)
        sentiment_ratios = {label: count / total_samples for label, count in sentiment_counts.items()}
        return {k: v for k, v in sorted(sentiment_ratios.items(), key=lambda item: item[1])}


    def predict_aspect(self, start:int, end:int, text:str=None):
        """
        Predicts the aspect of a given text or a range of texts using a pre-trained model.

        Args:
        - start (int): The starting index of the text range to analyze.
        - end (int): The ending index of the text range to analyze.
        - text (str, optional): The text to analyze. If None, the method will use the text range specified by start and end.

        Returns:
        - A dictionary containing the aspect ratios of the analyzed text(s). The keys are the aspect labels and the values are the corresponding ratios.
        """
        
        model_url = 'nlp-chula/aspect-finnlp-th'
        tokenizer = AutoTokenizer.from_pretrained(model_url, max_length = 412)
        tokenizer_kwargs = {'truncation':True,'max_length':412}
        aspect_classifier = pipeline(task='text-classification',
                                            tokenizer=tokenizer,
                                            model = model_url, 
                                            truncation =True)
        if text is None:
            self.aspect_prob = aspect_classifier(self.extract_df['paragraph'][start:end], **tokenizer_kwargs)
        else:
            self.aspect_prob = aspect_classifier(text, **tokenizer_kwargs)
        
        aspect_list = ['Profit/Loss', 'Financing', 'Product/Service', 'Economics',
       'Political', 'Environment', 'Investment', 'Social&People', 'Brand',
       'Governance', 'Technology', 'Others', 'Legal', 'Dividend', 'M&A', 'Rating']
        aspect_counts = {asp: 0 for asp in aspect_list}

        # Counting occurrences of each sentiment type
        for data in self.aspect_prob:
            aspect_counts[data['label']] += 1

        # Calculating sentiment ratios
        total_samples = len(self.aspect_prob)
        aspect_ratios = {label: count / total_samples for label, count in aspect_counts.items()}
        return {k: v for k, v in sorted(aspect_ratios.items(), key=lambda item: item[1])}

