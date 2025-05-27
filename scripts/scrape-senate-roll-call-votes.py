import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import csv
import time
import os
from datetime import datetime

def scrape_senate_votes_session(congress, session):
    """
    Scrapes Senate roll call votes for a specific congress and session.

    Args:
        congress: Congress number (e.g., 119, 118, 117)
        session: Session number (1 or 2)

    Returns:
        List of lists with vote information
    """

    url = f"https://www.senate.gov/legislative/LIS/roll_call_lists/vote_menu_{congress}_{session}.htm"

    # Send GET request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        print(f"Scraping Congress {congress}, Session {session}...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the year from the header (e.g., "(1989)")
    header_section = soup.find('section', id='legislative_header')
    header_h1 = header_section.find('h1') if header_section else soup.find('h1')
    header_text = header_h1.get_text(strip=True) if header_h1 else ""
    year_match = re.search(r'\((\d{4})\)', header_text)
    year = year_match.group(1) if year_match else ""

    # Find the votes table
    votes_table = soup.find('table', id='listOfVotes')
    if not votes_table:
        print(f"Could not find the votes table for Congress {congress}, Session {session}")
        return []

    # Find all rows (excluding header)
    rows = votes_table.find('tbody').find_all('tr') if votes_table.find('tbody') else votes_table.find_all('tr')[1:]

    votes_data = []

    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 5:
            continue

        # Extract vote number and URL
        vote_cell = cells[0]
        vote_link = vote_cell.find('a')
        if vote_link:
            vote_text = vote_link.get_text(strip=True).replace('\xa0', ' ')
            vote = vote_text
            vote_url = urljoin(url, vote_link.get('href', ''))
        else:
            vote = vote_cell.get_text(strip=True).replace('\xa0', ' ')
            vote_url = ""

        # Extract result
        result = cells[1].get_text(strip=True)

        # Extract description
        description = cells[2].get_text(strip=True)

        # Extract issue and URL
        issue_cell = cells[3]
        issue_link = issue_cell.find('a')
        if issue_link:
            issue = issue_link.get_text(strip=True)
            issue_url = issue_link.get('href', '')
            # Handle relative URLs
            if issue_url and not issue_url.startswith('http'):
                issue_url = urljoin('https://www.congress.gov/', issue_url) if issue_url.startswith('/') else issue_url
        else:
            issue = issue_cell.get_text(strip=True)
            issue_url = ""

        # Extract and reformat date
        raw_date = cells[4].get_text(strip=True).replace('\xa0', ' ')
        if year:
            try:
                # Parse e.g. "May 22 1989" or "Feb 18 1989"
                dt = datetime.strptime(f"{raw_date} {year}", "%b %d %Y")
                formatted_date = dt.strftime("%m/%d/%Y")
            except ValueError:
                # Fallback if parsing fails
                formatted_date = raw_date
        else:
            formatted_date = raw_date

        # Create vote record as list
        vote_record = [
            congress,
            session,
            vote,
            vote_url,
            result,
            description,
            issue,
            issue_url,
            formatted_date
        ]

        votes_data.append(vote_record)

    print(f"Found {len(votes_data)} votes for Congress {congress}, Session {session}")
    return votes_data


def scrape_all_senate_votes(start_congress=101, end_congress=119):
    """
    Scrapes Senate roll call votes from all congresses and sessions.

    Args:
        start_congress: Starting congress number (default 101 for 1989)
        end_congress: Ending congress number (default 119 for current)

    Returns:
        List of all vote lists from all sessions
    """

    all_votes = []

    for congress in range(start_congress, end_congress + 1):
        # Each congress has 2 sessions
        for session in [1, 2]:
            session_votes = scrape_senate_votes_session(congress, session)
            all_votes.extend(session_votes)

            # Add a small delay to be respectful to the server
            time.sleep(0.1)

    return all_votes


def save_to_csv(data, filename="../data/senate-roll-call-votes/senate-roll-call-votes.csv"):
    """Save the scraped data to a CSV file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Define CSV headers
    headers = [
        'congress',
        'session',
        'vote',
        'vote_url',
        'result',
        'description',
        'issue',
        'issue_url',
        'date'
    ]

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header row
        writer.writerow(headers)

        # Write data rows
        writer.writerows(data)

    print(f"Data saved to {filename}")


def main():
    """Main function to run the scraper"""
    print("Scraping Senate roll call votes from 101st Congress (1989) to 119th Congress (2025)...")

    votes = scrape_all_senate_votes()

    if votes:
        print(f"\nSuccessfully scraped {len(votes)} total votes")

        # Save to CSV file
        save_to_csv(votes)

        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total votes scraped: {len(votes)}")

        # Count by congress
        congress_counts = {}
        for vote in votes:
            congress = vote[0]  # congress is first element in list
            congress_counts[congress] = congress_counts.get(congress, 0) + 1

        print("\nVotes by Congress:")
        for cong in sorted(congress_counts.keys()):
            print(f"  {cong}th Congress: {congress_counts[cong]} votes")

        # Count results
        results = {}
        for vote in votes:
            result = vote[4]  # result is 5th element in list
            results[result] = results.get(result, 0) + 1

        print("\nResults breakdown:")
        for res, count in sorted(results.items()):
            print(f"  {res}: {count}")

    else:
        print("No votes were scraped")


if __name__ == "__main__":
    main()
