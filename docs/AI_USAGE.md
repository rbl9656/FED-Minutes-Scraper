Prompt 1: Help me create a web scraper for the meeting minutes for the FED from the official website: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm. This scraper will be in the form of three files shown below: 

# 1. Data Quality Assurance

- Validation rules for scraped data

# 2. Respectful Scraping

- Implement exponential backoff 

- Retry limit

# 3. Business Logic

- Basic data transformation pipeline

- Value-added calculations or insights

- Export in JSON format to a folder named data


Prompt 2: Edit these python files so that it is entirely self contained and does not need a demo file.

Prompt 3: Help me create a brief technical design file for this project.

AI vs. Human Written Code:

The majority of this code was generated using Claude by Anthropic. The first prompts used above created the format for the code and the second prompt focused it and edited the code so that it conformed to the format required. The group went through and reviewed the code line by line in order to make sure that it made sense. Edits were made to the paths in scraper.py so that the JSON files would be exported correctly to the data folder. Additionally, the test validation used in validators.py were edited for size as the original was too short. There were also added terms to the validator to expand the pool of applicable information. Edits were also made to the transformers.py terms for the same reason.

Bugs found in AI:

The main bug found in the AI written code was that the JSON files would not export to the correct folder. The code was written to export them to project folder, rather than the specifically named data folder. After going line by line, the code was added to line 80 of scraper.py so that the directory could be saved and used in future code to place the JSON and txt files in the data folder.