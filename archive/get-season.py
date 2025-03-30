import requests
import pandas as pd
import time
import os
from bs4 import BeautifulSoup
import re

def get_html(url):
    """
    Get HTML content from a URL with headers that mimic a browser
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.sports-reference.com/',
        'Connection': 'keep-alive',
    }
    
    print(f"Requesting URL: {url}")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to get {url}, status code: {response.status_code}")
        return None

def get_season_urls():
    """
    Get all season URLs from the main page, going back to 2000
    """
    url = 'https://www.sports-reference.com/cbb/seasons/'
    html = get_html(url)
    
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    seasons = []
    
    # Find the men's seasons table
    table = soup.find('table', id='seasons_NCAAM')
    if not table:
        print("Could not find men's seasons table")
        return []
    
    # Extract links to each season
    for row in table.find('tbody').find_all('tr'):
        season_cell = row.find('td', {'data-stat': 'season'})
        if not season_cell:
            # Try finding it as a th instead of td
            season_cell = row.find('th', {'data-stat': 'season'})
            
        if season_cell and season_cell.find('a'):
            link = season_cell.find('a')['href']
            season_url = f"https://www.sports-reference.com{link}"
            season_name = season_cell.text.strip()
            
            # Extract year from URL using regex
            year_match = re.search(r'/men/(\d{4})\.html', link)
            if year_match:
                year = year_match.group(1)
                # Only include seasons from 2000 onwards
                if int(year) >= 2000:
                    seasons.append({
                        'season_name': season_name,
                        'season_url': season_url,
                        'year': year,
                        'stats_url': f"https://www.sports-reference.com/cbb/seasons/men/{year}-school-stats.html"
                    })
    
    # Sort seasons by year (newest first)
    seasons.sort(key=lambda x: x['year'], reverse=True)
    print(f"Found {len(seasons)} seasons from 2000 onwards")
    return seasons

def clean_column_names(headers):
    """
    Clean up column names to have consistent naming
    """
    # Map of column name replacements
    name_map = {
        # Common column name issues
        'School': 'School',
        'Rk': 'Rank',
        'G': 'Games',
        'W': 'Wins',
        'L': 'Losses',
        'W-L%': 'Win_Pct',
        'SRS': 'SRS',
        'SOS': 'SOS',
        'Tm.': 'Points_For',
        'Opp.': 'Points_Against',
        'MP': 'Minutes_Played',
        'FG': 'FG',
        'FGA': 'FGA',
        'FG%': 'FG_Pct',
        '3P': '3P',
        '3PA': '3PA',
        '3P%': '3P_Pct',
        'FT': 'FT',
        'FTA': 'FTA',
        'FT%': 'FT_Pct',
        'ORB': 'ORB',
        'TRB': 'TRB',
        'AST': 'AST',
        'STL': 'STL',
        'BLK': 'BLK',
        'TOV': 'TOV',
        'PF': 'PF'
    }
    
    # Clean headers
    clean_headers = []
    for i, header in enumerate(headers):
        # For columns with duplicate or unclear names
        if header in ['DUMMY', '', None]:
            # Check if it's one of the conference/home/away columns
            if i == 8:  # First DUMMY column
                clean_headers.append('Conf_W')
            elif i == 10:  # Second DUMMY column
                clean_headers.append('Conf_L')
            elif i == 12:  # Third DUMMY column
                clean_headers.append('Home_W')
            elif i == 14:  # Fourth DUMMY column
                clean_headers.append('Home_L')
            elif i == 16:  # Fifth DUMMY column
                clean_headers.append('Away_W')
            elif i == 18:  # Sixth DUMMY column
                clean_headers.append('Away_L')
            elif i == 20:  # Seventh DUMMY column
                clean_headers.append('Points_Dummy')
            else:
                clean_headers.append(f'Column_{i}')
        else:
            # Use the mapping if available, otherwise use the original
            clean_headers.append(name_map.get(header, header))
    
    return clean_headers

def clean_data(df):
    """
    Clean the dataframe data
    """
    # Remove rows that are repeating headers
    if 'School' in df.columns:
        df = df[~df['School'].str.contains('School', na=False)]
    
    # Remove rows with NaN School
    df = df.dropna(subset=['School'])
    
    # Clean up numeric columns - only convert columns that actually exist
    possible_numeric_cols = [
        'Games', 'Wins', 'Losses', 'Win_Pct', 'SRS', 'SOS',
        'Conf_W', 'Conf_L', 'Home_W', 'Home_L', 'Away_W', 'Away_L',
        'Points_For', 'Points_Against',
        'FG', 'FGA', 'FG_Pct', '3P', '3PA', '3P_Pct',
        'FT', 'FTA', 'FT_Pct', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF'
    ]
    
    for col in possible_numeric_cols:
        if col in df.columns:
            try:
                # Convert to numeric, coerce errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not convert column {col} to numeric. Error: {e}")
    
    # Handle NCAA indicator in school names
    if 'School' in df.columns:
        # Create a new column for NCAA tournament teams
        df['NCAA_Tournament'] = df['School'].str.contains('NCAA', na=False)
        # Remove the NCAA indicator from school names
        df['School'] = df['School'].str.replace(' NCAA', '', regex=False)
    
    return df

def scrape_school_stats(stats_url, year):
    """
    Scrape the school stats table for a specific season
    """
    html = get_html(stats_url)
    
    if not html:
        return None
    
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', id='basic_school_stats')
    
    if not table:
        print(f"Could not find school stats table for {year}")
        return None
    
    # Manually extract table data
    data = []
    headers = []
    
    # First get the headers
    header_rows = table.find('thead').find_all('tr')
    if not header_rows:
        print(f"No header rows found for {year}")
        return None
        
    header_row = header_rows[-1]  # Use the last header row
    for th in header_row.find_all('th'):
        header_text = th.text.strip()
        # If header text is empty, use the data-stat attribute
        if not header_text and th.get('data-stat'):
            header_text = th.get('data-stat')
        headers.append(header_text)
    
    # Clean up the headers
    clean_headers = clean_column_names(headers)
    
    # Then get the data rows
    tbody = table.find('tbody')
    if not tbody:
        print(f"No table body found for {year}")
        return None
        
    for row in tbody.find_all('tr'):
        # Skip over header rows that might be in the tbody
        if 'class' in row.attrs and 'thead' in row.attrs.get('class', []):
            continue
            
        # Also skip rows with th that are header labels
        if row.find('th', scope='col'):
            continue
            
        row_data = []
        cells = row.find_all(['th', 'td'])
        
        # Only process rows that look like data (have enough cells)
        if len(cells) > 5:  # Arbitrary threshold
            for cell in cells:
                row_data.append(cell.text.strip())
            
            # Ensure the row has the right number of columns
            while len(row_data) < len(clean_headers):
                row_data.append('')
                
            if len(row_data) > len(clean_headers):
                row_data = row_data[:len(clean_headers)]
                
            data.append(row_data)
    
    # Check if we found any data
    if not data:
        print(f"No data rows found for {year}")
        return None
        
    # Create the dataframe
    df = pd.DataFrame(data, columns=clean_headers)
    
    # Add season year to the dataframe
    df['Season'] = year
    
    # Clean the data
    try:
        df = clean_data(df)
    except Exception as e:
        print(f"Error cleaning data for {year}: {e}")
        # Still return the uncleaned data
    
    return df

def save_dataframe(df, filename):
    """
    Save a dataframe to CSV
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")

def main():
    # Step 1: Get list of season URLs from 2000 onwards
    seasons = get_season_urls()
    
    # Save seasons info
    seasons_df = pd.DataFrame(seasons)
    save_dataframe(seasons_df, "data/seasons.csv")
    
    all_school_stats = []
    
    # Step 2: For each season, get school stats
    for season in seasons:
        print(f"Processing {season['season_name']} ({season['year']})")
        
        # Step 3: Scrape school stats
        try:
            stats_df = scrape_school_stats(season['stats_url'], season['year'])
            
            if stats_df is not None and not stats_df.empty:
                # Save individual season stats
                filename = f"data/school_stats_{season['year']}.csv"
                save_dataframe(stats_df, filename)
                
                # Add to list for combined file
                all_school_stats.append(stats_df)
            else:
                print(f"No data found for {season['year']}")
        except Exception as e:
            print(f"Error processing {season['year']}: {e}")
        
        # Be nice to the server
        time.sleep(2)
    
    # Step 4: Combine all stats into one file
    if all_school_stats:
        try:
            combined_df = pd.concat(all_school_stats, ignore_index=True)
            save_dataframe(combined_df, "data/all_school_stats.csv")
            print(f"Combined data for {len(all_school_stats)} seasons (2000-present)")
        except Exception as e:
            print(f"Error combining all stats: {e}")
    else:
        print("No data to combine")

if __name__ == "__main__":
    main()