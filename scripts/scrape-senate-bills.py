#!/usr/bin/env python3
"""
Senate Bill Text Scraper using Selenium

This script uses Selenium with a real browser to scrape bill text from congress.gov
"""

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import os
import time
import re
import random
from pathlib import Path

def clean_filename(filename):
    """Clean filename to be filesystem-safe"""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.strip().rstrip('.')
    return filename

def setup_driver():
    """Set up Chrome WebDriver with options to appear more human-like"""
    chrome_options = Options()
    
    # Basic options
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Make it look more like a regular browser
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--start-maximized')
    
    # Disable images and CSS for faster loading (optional)
    # chrome_options.add_argument('--disable-images')
    # chrome_options.add_argument('--disable-css')
    
    # Uncomment the next line if you want to run headless (no browser window)
    # chrome_options.add_argument('--headless')
    
    try:
        # Try to create driver (assumes chromedriver is in PATH)
        driver = webdriver.Chrome(options=chrome_options)
        
        # Execute script to hide automation indicators
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver
        
    except Exception as e:
        print(f"Error setting up Chrome driver: {e}")
        print("Make sure you have Chrome and chromedriver installed")
        print("You can download chromedriver from: https://chromedriver.chromium.org/")
        raise

def extract_bill_summary_content(driver, issue_name):
    """Extract content from bill-summary element"""
    try:
        bill_summary = driver.find_element(By.ID, "bill-summary")
        if bill_summary.is_displayed():
            children = bill_summary.find_elements(By.XPATH, "./*")
            if len(children) >= 3:
                text_parts = []
                for i in range(2, len(children)):  # Start from index 2 (third child)
                    child_text = children[i].text.strip()
                    if child_text:
                        text_parts.append(child_text)
                
                if text_parts:
                    content = "\n\n".join(text_parts)
                    print(f"Found content from {len(text_parts)} children (starting from 3rd) of bill-summary")
                    return content
                else:
                    print(f"No text content in children 3+ of bill-summary")
                    return ""
            else:
                print(f"Bill-summary has less than 3 children ({len(children)})")
                return ""
    except NoSuchElementException:
        # Try by class name
        try:
            bill_summary = driver.find_element(By.CLASS_NAME, "bill-summary")
            if bill_summary.is_displayed():
                children = bill_summary.find_elements(By.XPATH, "./*")
                if len(children) >= 3:
                    text_parts = []
                    for i in range(2, len(children)):
                        child_text = children[i].text.strip()
                        if child_text:
                            text_parts.append(child_text)
                    
                    if text_parts:
                        content = "\n\n".join(text_parts)
                        print(f"Found content from {len(text_parts)} children (starting from 3rd) of bill-summary (class)")
                        return content
                    else:
                        print(f"No text content in children 3+ of bill-summary (class)")
                        return ""
                else:
                    print(f"Bill-summary (class) has less than 3 children ({len(children)})")
                    return ""
        except NoSuchElementException:
            print(f"No bill-summary found")
            return None
    except Exception as e:
        print(f"Error extracting from bill-summary: {e}")
        return None
    
    return None

def scrape_bill_text_selenium(issue_url, issue_name, driver):
    """
    Scrape bill text using Selenium, specifically targeting children 3+ of bill-summary
    
    Args:
        issue_url (str): Base URL for the bill
        issue_name (str): Name of the bill/issue
        driver: Selenium WebDriver instance
    
    Returns:
        str: Text content of the bill, or empty string if no content
    """
    # Try /text first
    try:
        text_url = f"{issue_url}/text"
        print(f"Trying to scrape {issue_name} from {text_url}")
        
        driver.get(text_url)
        time.sleep(random.uniform(2, 4))
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Try to extract content
        content = extract_bill_summary_content(driver, issue_name)
        
        if content is not None:  # Found bill-summary
            if content:  # Has content
                return content
            else:  # Empty content, try /text/enr
                print(f"Found empty bill-summary on /text, trying /text/enr for {issue_name}")
                try:
                    enr_url = f"{issue_url}/text/enr"
                    driver.get(enr_url)
                    time.sleep(random.uniform(2, 4))
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                    
                    enr_content = extract_bill_summary_content(driver, f"{issue_name} (enr)")
                    if enr_content is not None:
                        if enr_content:
                            print(f"Found content on /text/enr page")
                            return enr_content
                        else:
                            print(f"No text content found on /text/enr page either")
                            return ""
                    
                except Exception as e:
                    print(f"Error trying /text/enr for {issue_name}: {e}")
                
                return ""  # Return empty if /text/enr didn't work
        
    except TimeoutException:
        print(f"Timeout loading {text_url}")
    except WebDriverException as e:
        if "403" in str(e) or "Forbidden" in str(e):
            print(f"403 Forbidden for {text_url}")
        else:
            print(f"WebDriver error for {text_url}: {e}")
    except Exception as e:
        print(f"Error scraping {text_url}: {e}")
    
    # Try other URL patterns if /text didn't work
    other_urls = [f"{issue_url}", f"{issue_url}/summary"]
    
    for url in other_urls:
        try:
            print(f"Trying fallback URL: {url}")
            driver.get(url)
            time.sleep(random.uniform(2, 4))
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            content = extract_bill_summary_content(driver, issue_name)
            if content is not None:
                return content if content else ""
                
        except Exception as e:
            print(f"Error with fallback URL {url}: {e}")
            continue
    
    # Last resort: try fallback selectors
    print("No bill-summary found, trying fallback selectors")
    fallback_selectors = [
        (By.CLASS_NAME, "overview"),
        (By.CLASS_NAME, "bill-text-container"),
        (By.CLASS_NAME, "generated-html-container"),
        (By.TAG_NAME, "main"),
        (By.CLASS_NAME, "main-wrapper"),
        (By.TAG_NAME, "article"),
        (By.CLASS_NAME, "content"),
    ]
    
    for by, selector in fallback_selectors:
        try:
            elements = driver.find_elements(by, selector)
            for element in elements:
                if element.is_displayed():
                    text = element.text.strip()
                    if text and len(text) > 100 and not is_boilerplate(text):
                        print(f"Found content using fallback {by}='{selector}'")
                        return text
                    
        except Exception as e:
            print(f"Error with fallback selector {by}='{selector}': {e}")
            continue
    
    # If absolutely nothing worked, return empty string
    print(f"No content found for {issue_name}, returning empty")
    return ""

def is_boilerplate(text):
    """Check if text is likely boilerplate/navigation content"""
    boilerplate_indicators = [
        'skip to main content',
        'navigation',
        'breadcrumb',
        'footer',
        'header',
        'menu',
        'search results',
        'javascript is disabled',
        'cookie policy',
        'privacy policy'
    ]
    
    text_lower = text.lower()
    short_text = len(text) < 500
    
    if short_text:
        for indicator in boilerplate_indicators:
            if indicator in text_lower:
                return True
    
    return False

def save_bill_text(text_content, issue_name, output_dir):
    """Save bill text to a file - just the raw content, no headers"""
    filename = clean_filename(f"{issue_name}.txt")
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            if text_content:
                f.write(text_content)
            # If no content, just create an empty file (don't write anything)
        
        print(f"Saved {issue_name} to {filepath} ({'with content' if text_content else 'empty'})")
        return True
        
    except Exception as e:
        print(f"Error saving {issue_name}: {e}")
        return False

def main():
    """Main function to orchestrate the scraping process"""
        
    # File paths
    csv_file = "../data/senate-roll-call-votes/senate-roll-call-votes.csv"
    output_dir = "../data/senate-bills"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    driver = None
    
    try:
        # Load the CSV file
        print(f"Loading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records")
        
        # Get unique bills to avoid duplicates
        unique_bills = df[['issue', 'issue_url', 'date']].drop_duplicates()
        print(f"Found {len(unique_bills)} unique items before filtering")
        
        # Filter to only include items where issue_url contains "bill"
        unique_bills = unique_bills[unique_bills['issue_url'].str.contains('bill', case=False, na=False)]
        print(f"Found {len(unique_bills)} unique bills after filtering (containing 'bill' in URL)")
        
        # Convert date column to datetime and sort by most recent first
        unique_bills['date'] = pd.to_datetime(unique_bills['date'])
        unique_bills = unique_bills.sort_values('date', ascending=False).reset_index(drop=True)
        print(f"Sorted bills by date (most recent first)")
        print(f"Date range: {unique_bills['date'].min().strftime('%Y-%m-%d')} to {unique_bills['date'].max().strftime('%Y-%m-%d')}")
        
        # Set up the WebDriver
        print("Setting up Chrome WebDriver...")
        driver = setup_driver()
        print("WebDriver setup successful")
        
        # Track progress
        success_count = 0
        error_count = 0
        skipped_count = 0
        empty_count = 0
        
        # Process bills
        for index, row in unique_bills.iterrows():
            issue_name = row['issue']
            issue_url = row['issue_url']
            
            # Skip if URL is missing or invalid
            if pd.isna(issue_url) or not issue_url.startswith('http'):
                print(f"Skipping {issue_name} - invalid URL: {issue_url}")
                skipped_count += 1
                continue
            
            # Check if file already exists
            filename = clean_filename(f"{issue_name}.txt")
            filepath = os.path.join(output_dir, filename)
            
            if os.path.exists(filepath):
                print(f"Skipping {issue_name} - file already exists")
                skipped_count += 1
                continue
            
            # Scrape the bill text (now always returns a string, even if empty)
            text_content = scrape_bill_text_selenium(issue_url, issue_name, driver)
            
            # Always try to save, even if content is empty
            if save_bill_text(text_content, issue_name, output_dir):
                success_count += 1
                if not text_content:
                    empty_count += 1
            else:
                error_count += 1
            
            # Random delay between requests (4-7 seconds)
            delay = random.uniform(4, 7)
            print(f"Waiting {delay:.1f} seconds before next request...")
            time.sleep(delay)
            
            # Progress update every 5 bills
            if (index + 1) % 5 == 0:
                print(f"Progress: {index + 1}/{len(unique_bills)} bills processed")
                print(f"Success: {success_count}, Errors: {error_count}, Empty: {empty_count}, Skipped: {skipped_count}")
                
                # Longer break every 15 bills to avoid detection
                if (index + 1) % 15 == 0:
                    break_time = 10
                    print(f"Taking a {break_time:.1f} second break...")
                    time.sleep(break_time)
        
        # Final summary
        print(f"Scraping completed!")
        print(f"Success: {success_count}")
        print(f"  - With content: {success_count - empty_count}")
        print(f"  - Empty bills: {empty_count}")
        print(f"Errors: {error_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Total processed: {success_count + error_count + skipped_count}")
        
    except FileNotFoundError:
        print(f"CSV file not found: {csv_file}")
    except KeyboardInterrupt:
        print("Scraping interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean up
        if driver:
            print("Closing WebDriver...")
            driver.quit()

if __name__ == "__main__":
    main()