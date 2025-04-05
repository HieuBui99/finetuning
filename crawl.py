import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import argparse

def download_file(url, output_directory):
    """
    Download a file from a URL and save it to the output directory.
    Returns the saved file path if successful, None otherwise.
    """
    # Get the filename from the URL
    filename = os.path.basename(urlparse(url).path)
    
    # Create the full output path
    output_path = os.path.join(output_directory, filename)
    
    try:
        # Make the request with a timeout
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for bad responses
        
        # Save the file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"Downloaded: {filename}")
        return output_path
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def find_pdf_links(url):
    """
    Find all PDF links on a webpage.
    Returns a list of absolute URLs to PDF files.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all anchor tags (links)
        links = soup.find_all('a')
        
        # Extract links that point to PDF files
        pdf_links = []
        for link in links:
            href = link.get('href')
            if href and href.lower().endswith('.pdf'):
                # Convert relative URL to absolute URL
                absolute_url = urljoin(url, href)
                pdf_links.append(absolute_url)
        
        return pdf_links
    
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return []

def crawl_website_for_pdfs(url, output_directory, recursive=False, visited=None):
    """
    Crawl a website for PDF files and download them.
    
    Args:
        url: The URL of the website to crawl
        output_directory: Directory to save downloaded PDFs
        recursive: Whether to recursively follow links on the same domain
        visited: Set of visited URLs (used for recursive crawling)
    """
    if visited is None:
        visited = set()
    
    # Skip if already visited
    if url in visited:
        return
    
    visited.add(url)
    base_domain = urlparse(url).netloc
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Find PDF links on the page
    pdf_links = find_pdf_links(url)
    
    # Download each PDF
    for pdf_url in pdf_links:
        download_file(pdf_url, output_directory)
    
    # If recursive, find and follow links on the same domain
    if recursive:
        try:
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links on the page
            links = soup.find_all('a')
            
            for link in links:
                href = link.get('href')
                if not href:
                    continue
                
                absolute_url = urljoin(url, href)
                link_domain = urlparse(absolute_url).netloc
                
                # Only follow links on the same domain
                if link_domain == base_domain and absolute_url not in visited:
                    crawl_website_for_pdfs(absolute_url, output_directory, recursive, visited)
        
        except Exception as e:
            print(f"Error processing {url}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download PDF files from a website')
    parser.add_argument('url', help='URL of the website to crawl for PDFs')
    parser.add_argument('-o', '--output', default='data/PDF', help='Directory to save downloaded PDFs')
    
    args = parser.parse_args()
    
    print(f"Starting to crawl {args.url} for PDF files...")
    crawl_website_for_pdfs(args.url, args.output)
    print(f"Finished! PDFs saved to {args.output}")

if __name__ == "__main__":
    main()