import sys
import requests
import re
import time
import os
import csv
from bs4 import BeautifulSoup

# Create a session to maintain cookies and state
session = requests.Session()

def fetch_senate_periodic_transaction_disclosures(batch_size=100):
    """
    Retrieves periodic transaction disclosures from the Senate's Electronic Financial Disclosure system.
    Handles authentication, CSRF tokens, and pagination.
    """
    
    # Initialize session and get CSRF token
    main_url = "https://efdsearch.senate.gov/search/"
    print("Accessing main search page to establish session...")
    main_response = session.get(main_url)
    
    if main_response.status_code != 200:
        print(f"Failed to access main page. Status code: {main_response.status_code}")
        return None
    
    # Extract CSRF token
    csrf_token = None
    csrf_pattern = re.compile(r'name="csrfmiddlewaretoken" value="([^"]+)"')
    match = csrf_pattern.search(main_response.text)
    
    if match:
        csrf_token = match.group(1)
        print(f"CSRF token obtained: {csrf_token[:5]}...{csrf_token[-5:]}")
    else:
        print("Failed to find CSRF token. Site may have changed.")
        return None
    
    # Accept terms and conditions
    print("Submitting agreement form...")
    agreement_url = "https://efdsearch.senate.gov/search/home/"
    agreement_data = {
        "csrfmiddlewaretoken": csrf_token,
        "prohibition_agreement": "1",
    }
    
    agreement_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/html,application/xhtml+xml,application/xml",
        "Referer": main_url,
    }
    
    agreement_response = session.post(agreement_url, headers=agreement_headers, data=agreement_data)
    
    if agreement_response.status_code != 200:
        print(f"Failed to submit agreement. Status code: {agreement_response.status_code}")
        return None
    
    # URL for the data request
    url = "https://efdsearch.senate.gov/search/report/data/"
    
    # Set up headers with the CSRF token
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://efdsearch.senate.gov",
        "Referer": "https://efdsearch.senate.gov/search/",
        "X-CSRFToken": csrf_token,
    }
    
    # Initialize variables for pagination
    start = 0
    records = []
    total_records = None
    
    # Loop to handle pagination
    while True:
        # Update the starting position in the form data
        form_data = {
            "draw": str(start // batch_size + 1),  # Increment draw number with each request
            "columns[0][data]": "0",
            "columns[0][name]": "",
            "columns[0][searchable]": "true",
            "columns[0][orderable]": "true",
            "columns[0][search][value]": "",
            "columns[0][search][regex]": "false",
            "columns[1][data]": "1",
            "columns[1][name]": "",
            "columns[1][searchable]": "true",
            "columns[1][orderable]": "true",
            "columns[1][search][value]": "",
            "columns[1][search][regex]": "false",
            "columns[2][data]": "2",
            "columns[2][name]": "",
            "columns[2][searchable]": "true",
            "columns[2][orderable]": "true",
            "columns[2][search][value]": "",
            "columns[2][search][regex]": "false",
            "columns[3][data]": "3",
            "columns[3][name]": "",
            "columns[3][searchable]": "true",
            "columns[3][orderable]": "true",
            "columns[3][search][value]": "",
            "columns[3][search][regex]": "false",
            "columns[4][data]": "4",
            "columns[4][name]": "",
            "columns[4][searchable]": "true",
            "columns[4][orderable]": "true",
            "columns[4][search][value]": "",
            "columns[4][search][regex]": "false",
            "order[0][column]": "4",
            "order[0][dir]": "desc",
            "start": str(start),  # Starting position for this batch
            "length": str(batch_size),  # Number of records per request
            "search[value]": "",
            "search[regex]": "false",
            "report_types": "[11]",
            "filer_types": "[]",
            "submitted_start_date": "01/01/2012 00:00:00",
            "submitted_end_date": "",
            "candidate_state": "",
            "senator_state": "",
            "office_id": "",
            "first_name": "",
            "last_name": "",
            "csrfmiddlewaretoken": csrf_token,
        }
        
        try:
            # Send the POST request using the session that contains our cookies
            print(f"Fetching records {start+1}-{start+batch_size}...")
            response = session.post(url, headers=headers, data=form_data)
            
            # Check if the request was successful
            if response.status_code == 200:
                batch_data = response.json()
                
                # Get total records count (only need to do this once)
                if total_records is None:
                    total_records = batch_data.get('recordsTotal', 0)
                    print(f"Total records available: {total_records}")
                
                # Add this batch to our combined data
                current_batch = batch_data.get('data', [])
                records.extend(current_batch)
                print(f"Retrieved batch with {len(current_batch)} records. Total collected: {len(records)}")
                
                # Check if we've reached the end or hit our max_records limit
                if not current_batch or len(records) >= total_records:
                    break
                
                # Move to the next batch
                start += batch_size
                
                # Add a small delay to be considerate to the server
                time.sleep(1)
                
            else:
                print(f"Failed to retrieve data. Status code: {response.status_code}")
                print(f"Response: {response.text[:500]}...")  # Showing first 500 chars
                break
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break
    
    return records

def process_and_filter_records(records):
    """
    Filters and cleans records, keeping only the latest amendments.
    """
    print(f"\nFiltering for each senator's periodic transaction disclosures...")

    filtered_records = {}
    
    # Process each record
    for record in records:
            
        first_name = record[0].strip().replace(',', '')
        last_name = record[1].strip().replace(',', '')
        full_name = f"{first_name} {last_name}"
        full_name_title = record[2].strip()
        report_link_html = record[3]
        date_filed = record[4]
        
        
        # Extract the report title and URL from the HTML link
        href_match = re.search(r'href="([^"]+)"', report_link_html)
        title_match = re.search(r'>([^<]+)<', report_link_html)
        amendment_match = re.search(r'\(Amendment (\d+)\)', report_link_html)
        date_match = re.search(r'for (\d{2}/\d{2}/\d{4})', report_link_html)
        
        if not href_match or not title_match:
            continue
            
        url = f"https://efdsearch.senate.gov{href_match.group(1)}"
        report_title = title_match.group(1).strip()
        report_date = date_match.group(1).strip()
        amendment = int(amendment_match.group(1)) if amendment_match else 0

        key = (full_name, report_date)

        if key not in filtered_records or filtered_records[key]["amendment"] < amendment:
            filtered_records[key] = {
                "fullName": full_name,
                "firstName": first_name,
                "lastName": last_name,
                "fullNameTitle": full_name_title,
                "dateFiled": date_filed,
                "fileName": report_title,
                "url": url,
                "amendment": amendment
            }
    
    print(f"Total periodic transation disclosures after filtering: {len(filtered_records)}")
    return list(filtered_records.values())


def extract_transactions(html_content):
    """
    Extracts transaction data from the HTML content of a periodic transaction disclosure.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the transactions table
    transactions_table = soup.select_one('section.card .table.table-striped')
    if not transactions_table:
        return []

    transactions = []
    for row in transactions_table.select('tbody tr'):
        columns = row.find_all('td')
        if len(columns) < 9:
            continue  # skip malformed rows

        transaction_date = columns[1].get_text(strip=True)
        owner = columns[2].get_text(strip=True)

        # Extract ticker (may be a link or plain text)
        ticker_cell = columns[3]
        ticker_link = ticker_cell.find('a')
        ticker = ticker_link.get_text(strip=True) if ticker_link else ticker_cell.get_text(strip=True)

        asset_name = " ".join(columns[4].get_text(strip=True).replace(',', ';').split())
        asset_type = columns[5].get_text(strip=True)
        transaction_type = columns[6].get_text(strip=True)
        amount = columns[7].get_text(strip=True)

        transactions.append({
            'Transaction Date': transaction_date,
            'Owner': owner,
            'Ticker': ticker,
            'Asset Name': asset_name,
            'Asset Type': asset_type,
            'Transaction Type': transaction_type,
            'Amount': amount,
        })
    
    return transactions

def download_transactions(records):
    """
    Downloads transaction data for each senator and saves to individual CSV files.
    """
    os.makedirs(os.path.join("..", "data/senator-transactions"), exist_ok=True)
    senators = set()
    for record in records:
        print(f"Processing {record['fullName']} {record['fileName']}...")
        url = record['url']

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        try:
            # Add a delay to avoid overloading the server
            time.sleep(0.2)
            
            # Make the request
            response = session.get(url, headers=headers)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Scrape transactions
            transactions = extract_transactions(response.text)
            
            # Skip saving if there are no transactions
            if not transactions:
                print(f"✗ No transactions found in {record['fullName']} {record['fileName']}. Skipping CSV creation.")
                continue

            csv_filename = os.path.join("..", "data/senator-transactions", f"{record['fullName']}.csv")
            csv_exists = record['fullName'] in senators

            # Delete pre-existing csv
            if not csv_exists and os.path.exists(csv_filename):
                os.remove(csv_filename)

            # Open in append mode
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Owner', 'Ticker', 'Asset Name', 'Asset Type', 'Transaction Type', 'Transaction Date', 'Amount']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not csv_exists:
                    writer.writeheader()
                    senators.add(record['fullName'])

                for transaction in transactions:
                    writer.writerow(transaction)
            
            print(f"✓ Saved {len(transactions)} transactions to {csv_filename}")
            
        except Exception as e:
            print(f"✗ Error processing {record['fullName']}: {str(e)}")

if __name__ == "__main__":
    records = fetch_senate_periodic_transaction_disclosures()
    filtered_records = process_and_filter_records(records)
    download_transactions(filtered_records)
