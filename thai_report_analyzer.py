from pypdf import PdfReader
import difflib
import kenlm
import pytesseract
import fitz
from PIL import Image
import re, os
import pandas as pd
from transformers import AutoTokenizer, pipeline

class ReportAnalyzer:

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
    
    def __init__(self):
        sentiment_model_url = 'nlp-chula/augment-sentiment-finnlp-th'
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_url, max_length=412)
        self.tokenizer_kwargs = {'truncation': True, 'max_length': 412}
        self.sentiment_classifier = pipeline(task='text-classification',
                                        tokenizer=sentiment_tokenizer,
                                        model = sentiment_model_url, 
                                        truncation =True)

        aspect_model_url = 'nlp-chula/aspect-finnlp-th'
        aspect_tokenizer = AutoTokenizer.from_pretrained(aspect_model_url, max_length=412)
        self.aspect_classifier = pipeline(task='text-classification',
                                        tokenizer=aspect_tokenizer,
                                        model=aspect_model_url,
                                        truncation=True)
        
        self.kenlm_model = kenlm.Model('kenlm/sixgram.arpa')
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
        for _, (logprob, _, _) in enumerate(self.kenlm_model.full_scores(text)):
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
        tone_list = ['','่','้','๊','๋'] # List of Thai tone marks.
        
        # Replace errors with correct counterparts using a predefined dictionary.
        for error, correction in self.correct_dict.items():
            text = text.replace(error, correction)
        
        for text_ind, char in enumerate(text): 
            text_list.append(char) # Append each character to the list.
            
             # Check for specific characters and store their indices.
            if char == 'า' or char == 'ำ':
                index = len(text_list)-1
                index_list.append(index)
            elif char in ['ั','ึ','ิ','ี','ื','ำ']:
                index = len(text_list)-1
                index_list.append(index)
                if (text_ind + 1 < len(text) and text[text_ind + 1] not in tone_list) or (text_ind + 1 == len(text)):
                  text_list.append('')
                
            
        for index in index_list:
            if text_list[index] == 'า' or text_list[index] == 'ำ':
                text_list[index] = 'า'
                word_ah = '\r'.join(text_list)
                text_list[index] = 'ำ'
                word_um = '\r'.join(text_list)

                # Choose the correct form based on spelling scores.
                if self.calc_spelling_score(word_ah) > self.calc_spelling_score(word_um):
                    text_list[index] = 'า'
                else:
                    text_list[index] = 'ำ'
            else:
                if len(text_list) < index+1:
                    pass
                else:
                    best_tone = max(tone_list, key=lambda tone: self.calc_spelling_score('\r'.join(text_list[:index+1] + [tone] + text_list[index+2:])))
                    text_list[index+1] = best_tone
                        
                text = ''.join(text_list) # Rejoin the text after modifications.
        return text
    
    def read_pdf_ocr(self, file_path):
        """
        Reads in a PDF file and performs OCR (Optical Character Recognition) on it to extract text.
        The function first reads in the PDF file and extracts text from each page. Then, it converts each page to a JPG image
        and performs OCR on the image to extract text. The extracted text is cleaned up and returned as a list of strings.
        
        Returns:
        - extracted_text: A list of strings, where each string represents a line of text extracted from the PDF.
        """
        
        # Step 1: Read in the PDF and extract text
        pdf_file = file_path
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

    def read_pdf(self, file_path):
        """
        Reads a PDF file and returns a list of cleaned text from each page.

        Returns:
        page_list (list): A list of cleaned text from each page of the PDF file.
        """
        page_list = []
        reader = PdfReader(file_path)
        for _, page in enumerate(reader.pages):
            text = page.extract_text()
            line_list = text.split('\n')
            for line_idx, line in enumerate(line_list):
                line_list[line_idx] = self.clean_text(line)
            line_list.pop()
            page_list += line_list # one-page
        return page_list
    
    def extract_section(self, page_list):
        """
        Extracts specific sections from a given page list.

        Args:
            page_list (list): A list of lines representing a page.

        Returns:
            dict: A dictionary containing extracted sections as values, with section names as keys.
        """
        result_dict = {}
        for line_idx, line in enumerate(page_list):
            if line == '' or re.match('^\s$',line): # clean blank space
                page_list.pop(line_idx)
        for i in range(1,4):
            if i == 1:
                key = 'risk'
                keyword_start = '2.การบริหารจัดการความเสี่ยง'
                keyword_end = '3.การขับเคลื่อนธุรกิจเพื่อความยั่งยืน'
            elif i == 2:
                key = 'sustainability'
                keyword_start = '3.การขับเคลื่อนธุรกิจเพื่อความยั่งยืน'
                keyword_end = '4.การวิเคราะห์และคำอธิบายของฝ่ายจัดการ'
            elif i == 3:
                key = 'md&a'
                keyword_start = 'การวิเคราะห์และคำอธิบายของฝ่ายจัดการ'
                keyword_end = 'การรับรองความถูกต้องของข้อมูล'
            check_start = difflib.get_close_matches(keyword_start, page_list)
            check_stop = difflib.get_close_matches(keyword_end, page_list)
            if len(check_start) != 0:
                start_index = page_list.index(check_start[0])
            else:
                start_index = 0
            if len(check_stop) != 0:
                stop_index = page_list.index(check_stop[0])
            else:
                stop_index = start_index + 500
            result_dict[key] = '\n'.join(page_list[start_index:stop_index])
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
        section_list = []
        para_list = []
        for section in result_dict.keys():
            for para in result_dict[section].split('\n'):
                section_list.append(section)
                para_list.append(para)
        df = pd.DataFrame({'section': section_list, 'paragraph': para_list})
        df['isTable'] = df['paragraph'].apply(lambda x: self.likely_table(str(x), threshold=20))
        df['isIncomplete'] = df['paragraph'].apply(lambda x: self.incomplete_sentence(str(x)))

        df = df[(df['isTable'] == 0) & (df['isIncomplete'] == 0)].reset_index()[['section', 'paragraph']]
        return df

    def analyze_text(self, text):
        """
        Analyzes the sentiment and aspect of a given text.

        Args:
        - text (str): The text to analyze.

        Returns:
        - A dictionary containing the sentiment ratios and aspect ratios of the analyzed text.
          The keys are 'Sentiment' and 'Aspect', and the values are dictionaries providing
          the corresponding ratios for each sentiment label and aspect label.
        """
        text_list = text.split('\n')

        # Predict Sentiment
        self.sentiment_prob = self.sentiment_classifier(text, **self.tokenizer_kwargs)
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

        # Counting occurrences of each sentiment type
        for data in self.sentiment_prob:
            sentiment_counts[data['label']] += 1

        # Calculating sentiment ratios
        total_samples = len(self.sentiment_prob)
        sentiment_ratios = {label: count / total_samples for label, count in sorted(sentiment_counts.items(), key=lambda item: item[1], reverse=True)}
    
        # Predict Aspect
        self.aspect_prob = self.aspect_classifier((text_list), **self.tokenizer_kwargs)

        aspect_list = ['Profit/Loss', 'Financing', 'Product/Service', 'Economics',
       'Political', 'Environment', 'Investment', 'Social&People', 'Brand',
       'Governance', 'Technology', 'Others', 'Legal', 'Dividend', 'M&A', 'Rating']
        aspect_counts = {asp: 0 for asp in aspect_list}

        # Counting occurrences of each aspect type
        for data in self.aspect_prob:
            aspect_counts[data['label']] += 1

        # Calculating aspect ratios
        total_samples = len(self.aspect_prob)
        aspect_ratios = {label: count / total_samples for label, count in sorted(aspect_counts.items(), key=lambda item: item[1], reverse=True)}

        return {'Sentiment': sentiment_ratios, 'Aspect': aspect_ratios}
    
    def analyze_file(self, file_path:str, file_type:str="pdf", start:int=None, end:int=None):
        """
        Analyzes the sentiment and aspect of text extracted from a file using pre-trained models.

        Args:
        - file_path (str): The path to the file for analysis.
        - file_type (str, optional): The type of the file ('pdf' or 'ocr'). Default is 'pdf'.
        - start (int, optional): The starting index of the text range to analyze.
        - end (int, optional): The ending index of the text range to analyze.

        Returns:
        - A dictionary containing the sentiment ratios and aspect ratios of the analyzed text.
        The keys are 'Sentiment' and 'Aspect', and the values are dictionaries providing
        the corresponding ratios for each sentiment label and aspect label.
        """
        print("Loading...")
        if file_type == 'pdf':
            line_list = self.read_pdf(file_path)
        elif file_type == 'ocr':
            line_list = self.read_pdf_ocr(file_path)
        section_dict = self.extract_section(line_list)
        self.extract_df = self.build_dataset(section_dict)

        # Predict Sentiment
        self.sentiment_prob = self.sentiment_classifier(list(self.extract_df['paragraph'][start:end]), **self.tokenizer_kwargs)
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

        # Counting occurrences of each sentiment type
        for data in self.sentiment_prob:
            sentiment_counts[data['label']] += 1

        # Calculating sentiment ratios
        total_samples = len(self.sentiment_prob)
        sentiment_ratios = {label: count / total_samples for label, count in sorted(sentiment_counts.items(), key=lambda item: item[1], reverse=True)}

        # Predict Aspect
        self.aspect_prob = self.aspect_classifier(list(self.extract_df['paragraph'][start:end]), **self.tokenizer_kwargs)
        aspect_list = ['Profit/Loss', 'Financing', 'Product/Service', 'Economics',
       'Political', 'Environment', 'Investment', 'Social&People', 'Brand',
       'Governance', 'Technology', 'Others', 'Legal', 'Dividend', 'M&A', 'Rating']
        aspect_counts = {asp: 0 for asp in aspect_list}

        # Counting occurrences of each aspect type
        for data in self.aspect_prob:
            aspect_counts[data['label']] += 1

        # Calculating aspect ratios
        total_samples = len(self.aspect_prob)
        aspect_ratios = {label: count / total_samples for label, count in sorted(aspect_counts.items(), key=lambda item: item[1], reverse=True)}

        return {'Sentiment': sentiment_ratios, 'Aspect': aspect_ratios}
