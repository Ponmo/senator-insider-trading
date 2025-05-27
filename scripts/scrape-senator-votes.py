import pandas as pd
import requests
import xml.etree.ElementTree as ET
import os
import time
import json
from collections import defaultdict
from urllib.parse import urlparse
import re


def get_xml_url_from_vote_url(vote_url):
    """Convert vote URL to XML URL format"""
    if not vote_url or pd.isna(vote_url):
        return None
    
    # Extract the XML URL by replacing .htm with .xml
    xml_url = vote_url.replace('.htm', '.xml')
    return xml_url


def fetch_xml_data(xml_url, max_retries=3, delay=1):
    """Fetch XML data from URL with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Fetching: {xml_url}")
            response = requests.get(xml_url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {xml_url}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                print(f"Failed to fetch {xml_url} after {max_retries} attempts")
                return None
    return None


def parse_xml_vote_data(xml_content):
    """Parse XML content and extract voting information"""
    try:
        root = ET.fromstring(xml_content)
        
        # Extract vote information for each member
        members_data = []
        members = root.findall('.//member')
        
        for member in members:
            first_name = member.find('first_name')
            last_name = member.find('last_name')
            party = member.find('party')
            state = member.find('state')
            vote_cast = member.find('vote_cast')
            
            if all(elem is not None for elem in [first_name, last_name, vote_cast]):
                full_name = f"{first_name.text} {last_name.text}"
                vote = vote_cast.text
                
                # Only include Yea/Nay votes, skip Not Voting, Present, etc.
                if vote in ['Yea', 'Nay']:
                    members_data.append({
                        'name': full_name,
                        'vote': vote,
                        'party': party.text if party is not None else None,
                        'state': state.text if state is not None else None
                    })
        
        return {
            'members': members_data
        }
        
    except ET.ParseError as e:
        print(f"Failed to parse XML: {str(e)}")
        return None


def build_senator_votes_dict(filtered_df, original_df, output_dir):
    """Build the main dictionary of senator votes with resume capability"""
    # Load existing results
    senator_votes, scraped_issues = load_existing_results(output_dir)
    
    # Use only the filtered DataFrame (which already has unique issues with latest dates)
    # Create unique identifier for each vote (issue + vote_number)
    filtered_df = filtered_df.copy()
    filtered_df['vote_id'] = filtered_df['issue'] + '_vote_' + filtered_df['vote'].str.extract(r'(\d+)')[0].astype(str)
    
    # Filter out already scraped vote IDs
    remaining_votes = filtered_df[~filtered_df['vote_id'].isin(scraped_issues)]
    
    total_filtered = len(filtered_df)
    already_scraped = len(scraped_issues.intersection(set(filtered_df['vote_id'])))
    remaining_to_process = len(remaining_votes)
    
    print(f"Total unique bills to process: {total_filtered}")
    print(f"Already scraped: {already_scraped}")
    print(f"Remaining to process: {remaining_to_process}")
    
    if remaining_to_process == 0:
        print("All bills have already been processed!")
        return senator_votes
    
    # Process each remaining row
    processed = 0
    
    try:
        for index, row in remaining_votes.iterrows():
            processed += 1
            current_vote_id = row['vote_id']
            current_issue = row['issue']
            progress = already_scraped + processed
            
            print(f"Progress: {progress}/{total_filtered} - Processing: {current_issue} (Vote {row['vote']})")
            
            # Get XML URL
            xml_url = get_xml_url_from_vote_url(row['vote_url'])
            if not xml_url:
                print(f"  Skipping: No valid vote URL")
                scraped_issues.add(current_vote_id)
                continue
            
            # Fetch and parse XML data
            xml_content = fetch_xml_data(xml_url)
            if not xml_content:
                print(f"  Skipping: Failed to fetch XML")
                scraped_issues.add(current_vote_id)
                continue
            
            vote_data = parse_xml_vote_data(xml_content)
            if not vote_data:
                print(f"  Skipping: Failed to parse XML")
                scraped_issues.add(current_vote_id)
                continue
            
            # Use the issue from CSV as the document number
            document_number = current_issue
            
            # Add votes to the dictionary
            votes_added = 0
            for member in vote_data['members']:
                senator_name = member['name']
                vote = member['vote']
                
                # Add document to the appropriate vote list (avoid duplicates)
                if document_number not in senator_votes[senator_name][vote]:
                    senator_votes[senator_name][vote].append(document_number)
                    votes_added += 1
            
            print(f"  Processed {len(vote_data['members'])} votes for document {document_number}")
            
            # Mark this vote as scraped
            scraped_issues.add(current_vote_id)
            
            # Save progress every 10 items
            if processed % 10 == 0:
                print(f"  Saving progress... ({progress}/{total_filtered})")
                save_progress(senator_votes, scraped_issues, output_dir)
    
    except KeyboardInterrupt:
        print(f"\nProcess interrupted! Saving progress...")
        save_progress(senator_votes, scraped_issues, output_dir)
        print(f"Progress saved. Processed {already_scraped + processed}/{total_filtered} votes.")
        print("You can restart the script to continue from where you left off.")
        raise
    
    # Final save
    save_progress(senator_votes, scraped_issues, output_dir)
    
    return senator_votes


def load_existing_results(output_dir):
    """Load existing results and scraped issues if they exist"""
    senator_votes_file = os.path.join(output_dir, "senator_votes.json")
    scraped_issues_file = os.path.join(output_dir, "scraped_issues.json")
    
    senator_votes = defaultdict(lambda: defaultdict(list))
    scraped_issues = set()
    
    # Load existing senator votes
    if os.path.exists(senator_votes_file):
        try:
            with open(senator_votes_file, 'r') as f:
                existing_votes = json.load(f)
                for senator, votes in existing_votes.items():
                    for vote_type, documents in votes.items():
                        senator_votes[senator][vote_type] = documents
            print(f"Loaded existing results for {len(existing_votes)} senators")
        except Exception as e:
            print(f"Warning: Could not load existing senator votes: {e}")
    
    # Load existing scraped issues
    if os.path.exists(scraped_issues_file):
        try:
            with open(scraped_issues_file, 'r') as f:
                scraped_issues = set(json.load(f))
            print(f"Found {len(scraped_issues)} previously scraped issues")
        except Exception as e:
            print(f"Warning: Could not load scraped issues: {e}")
    
    return senator_votes, scraped_issues


def save_progress(senator_votes, scraped_issues, output_dir):
    """Save current progress to files"""
    # Save senator votes
    senator_votes_file = os.path.join(output_dir, "senator_votes.json")
    result = {}
    for senator, votes in senator_votes.items():
        result[senator] = dict(votes)
    
    with open(senator_votes_file, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    
    # Save scraped issues
    scraped_issues_file = os.path.join(output_dir, "scraped_issues.json")
    with open(scraped_issues_file, 'w') as f:
        json.dump(list(scraped_issues), f, indent=2, sort_keys=True)
def save_results(senator_votes, scraped_issues, output_dir):
    """Save the final results to files"""
    # Save progress first
    save_progress(senator_votes, scraped_issues, output_dir)
    
    # Convert defaultdict to regular dict for final output
    result = {}
    for senator, votes in senator_votes.items():
        result[senator] = dict(votes)
    
    output_file = os.path.join(output_dir, "senator_votes.json")
    print(f"Final results saved to: {output_file}")
    
    # Also save a summary
    summary_file = os.path.join(output_dir, "senator_votes_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Senator Voting Summary\n")
        f.write(f"=====================\n\n")
        f.write(f"Total senators: {len(result)}\n")
        
        for senator, votes in sorted(result.items()):
            yea_count = len(votes.get('Yea', []))
            nay_count = len(votes.get('Nay', []))
            f.write(f"{senator}:\n")
            f.write(f"  Yea votes: {yea_count}\n")
            f.write(f"  Nay votes: {nay_count}\n")
            f.write(f"  Total votes: {yea_count + nay_count}\n\n")
    
    print(f"Summary saved to: {summary_file}")


def main():
    """Main function to orchestrate the scraping process"""
    # File paths
    csv_file = "../data/senate-roll-call-votes/senate-roll-call-votes.csv"
    output_dir = "../data/senator-votes"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the CSV file
        print(f"Loading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records")
        
        # Filter to only include items where issue_url contains "bill"
        bills_only = df[df['issue_url'].str.contains('bill', case=False, na=False)]
        print(f"Found {len(bills_only)} records with 'bill' in URL")
        
        # Convert date column to datetime
        bills_only = bills_only.copy()
        bills_only['date'] = pd.to_datetime(bills_only['date'])
        
        # Sort by date descending to keep latest duplicates
        bills_only = bills_only.sort_values('date', ascending=False)
        
        # Remove duplicates by issue name, keeping the first (most recent) occurrence
        unique_bills = bills_only.drop_duplicates(subset=['issue'], keep='first').reset_index(drop=True)
        print(f"Found {len(unique_bills)} unique bills after removing duplicates by issue name")
        print(f"Date range: {unique_bills['date'].min().strftime('%Y-%m-%d')} to {unique_bills['date'].max().strftime('%Y-%m-%d')}")
        
        # Build the senator votes dictionary using only the filtered unique bills
        print("Starting to build senator votes dictionary...")
        senator_votes = build_senator_votes_dict(unique_bills, df, output_dir)
        
        print(f"\nCompleted processing!")
        print(f"Total senators tracked: {len(senator_votes)}")
        
        # Display a sample of the results
        print("\nSample results:")
        for i, (senator, votes) in enumerate(list(senator_votes.items())[:3]):
            yea_count = len(votes.get('Yea', []))
            nay_count = len(votes.get('Nay', []))
            print(f"{senator}: {yea_count} Yea votes, {nay_count} Nay votes")
        
        # Save results
        save_results(senator_votes, set(), output_dir)
        
        print(f"\nScript completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        print("Please ensure the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()