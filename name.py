from bs4 import BeautifulSoup
import pandas as pd

# Load the HTML content from the file
file_path = 'D:/QAT_stock/page_source.html'
with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Extract company names and ticker symbols
companies = []
for company_info in soup.find_all('div', class_='info-card-content'):
    company_code = company_info.find('p', class_='companyCode').text.strip()
    company_name = company_info.find('h3', class_='companyName').text.strip()
    companies.append({'Company Name': company_name, 'Ticker Symbol': company_code})

# Create a DataFrame from the extracted data
companies_df = pd.DataFrame(companies)

# Save the data to a CSV file
companies_df.to_csv('D:/QAT_stock/company_names_and_tickers.csv', index=False)

print("Company names and ticker symbols have been successfully extracted and saved to company_names_and_tickers.csv")
