"""
Federal Reserve FOMC Minutes Web Scraper - Main Implementation
Implements respectful scraping with exponential backoff and comprehensive error handling
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import logging
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional, Tuple
import json
import os
import sys
from dataclasses import dataclass, asdict

# Add src directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from validators import FEDDataValidator
from transformers import FEDDataTransformer

@dataclass
class FOMCDocument:
    """Data structure for FOMC document metadata"""
    meeting_date: str
    release_date: str
    document_type: str  # 'minutes', 'statement', etc.
    html_url: Optional[str]
    pdf_url: Optional[str]
    content: Optional[str] = None
    scraped_at: Optional[str] = None
    validation_status: Optional[str] = None

class FEDScraper:
    """
    Federal Reserve FOMC Minutes Web Scraper
    Implements respectful scraping practices with comprehensive error handling
    """
    
    def __init__(self):
        self.base_url = "https://www.federalreserve.gov"
        self.calendar_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        
        # Respectful scraping configuration
        self.request_delay = 5  # Seconds between requests
        self.max_retries = 3
        self.retry_backoff_factor = 2
        self.timeout = 30
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        # Initialize validator and transformer
        self.validator = FEDDataValidator()
        self.transformer = FEDDataTransformer()
        
        # Setup logging
        self._setup_logging()
        
        # Create data directory
        self._ensure_data_directory()
    
    def _setup_logging(self):
        """Configure logging for the scraper"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fed_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
    
    def _get_session(self) -> requests.Session:
        """Create a configured requests session"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        return session
    
    def _respectful_request(self, url: str, session: Optional[requests.Session] = None) -> Optional[requests.Response]:
        """
        Make HTTP request with exponential backoff retry logic
        Implements respectful scraping practices
        """
        if session is None:
            session = self._get_session()
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Requesting: {url} (attempt {attempt + 1})")
                
                # Respectful delay between requests
                if attempt > 0:
                    delay = self.request_delay * (self.retry_backoff_factor ** (attempt - 1))
                    self.logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    time.sleep(self.request_delay)
                
                response = session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                self.logger.info(f"Successfully fetched: {url}")
                return response
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"All retry attempts failed for: {url}")
                    return None
        
        return None
    
    def extract_calendar_data(self) -> List[FOMCDocument]:
        """
        Extract FOMC meeting calendar data and document links
        Returns list of FOMCDocument objects with metadata
        """
        self.logger.info("Starting calendar data extraction...")
        
        session = self._get_session()
        response = self._respectful_request(self.calendar_url, session)
        
        if not response:
            self.logger.error("Failed to fetch calendar page")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        documents = []
        
        # Look for direct links to minutes
        minute_links = soup.find_all('a', href=re.compile(r'fomcminutes\d+'))
        
        for link in minute_links:
            doc = self._parse_minute_link(link)
            if doc and doc.html_url:
                documents.append(doc)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_documents = []
        for doc in documents:
            if doc.html_url not in seen_urls:
                seen_urls.add(doc.html_url)
                unique_documents.append(doc)
        
        self.logger.info(f"Extracted {len(unique_documents)} unique document references")
        return unique_documents
    
    def _parse_minute_link(self, link_element) -> Optional[FOMCDocument]:
        """Parse a minute link element to extract document metadata"""
        href = link_element.get('href')
        if not href:
            return None
        
        full_url = urljoin(self.base_url, href)
        
        # Extract date from URL
        date_match = re.search(r'fomcminutes(\d{8})', href)
        meeting_date = None
        if date_match:
            date_str = date_match.group(1)
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                meeting_date = date_obj.strftime('%B %d, %Y')
            except ValueError:
                pass
        
        # Try to find release date in surrounding text
        release_date = None
        parent = link_element.parent
        if parent:
            release_match = re.search(r'Released\s+(\w+\s+\d{1,2},\s+\d{4})', parent.get_text())
            if release_match:
                release_date = release_match.group(1)
        
        return FOMCDocument(
            meeting_date=meeting_date,
            release_date=release_date,
            document_type='minutes',
            html_url=full_url,
            pdf_url=None
        )
    
    def scrape_document_content(self, document: FOMCDocument) -> FOMCDocument:
        """
        Scrape the full content of a FOMC document
        Implements validation and content extraction
        """
        if not document.html_url:
            self.logger.warning("No HTML URL provided for document")
            return document
        
        self.logger.info(f"Scraping content from: {document.html_url}")
        
        session = self._get_session()
        response = self._respectful_request(document.html_url, session)
        
        if not response:
            document.validation_status = "FAILED_TO_FETCH"
            return document
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract main content - FOMC minutes are typically in specific containers
        content_selectors = [
            '.col-xs-12.col-sm-8.col-md-8',  # Common Fed layout
            '#article',
            '.feds-content',
            'main',
            '#content'
        ]
        
        content = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                content = content_element.get_text(strip=True, separator='\n')
                break
        
        # Fallback: get all paragraph text
        if not content:
            paragraphs = soup.find_all('p')
            content = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        # Final fallback: body text
        if not content:
            body = soup.find('body')
            if body:
                content = body.get_text(strip=True, separator='\n')
        
        document.content = content
        document.scraped_at = datetime.now().isoformat()
        
        # Validate the scraped content
        validation_result = self.validator.validate_document(document)
        document.validation_status = validation_result['status']
        
        self.logger.info(f"Content extracted: {len(content) if content else 0} characters")
        self.logger.info(f"Validation status: {document.validation_status}")
        
        return document
    
    def scrape_recent_minutes(self, limit: int = 3) -> List[FOMCDocument]:
        """
        Scrape the most recent FOMC minutes
        Main entry point for getting current data
        """
        self.logger.info(f"Starting scrape of {limit} most recent FOMC minutes")
        
        # Get calendar data
        documents = self.extract_calendar_data()
        
        # Filter for minutes only and sort by URL (which contains dates)
        minute_docs = [doc for doc in documents if doc.document_type == 'minutes' and doc.html_url]
        
        # Sort by URL to get most recent (URLs contain dates)
        minute_docs.sort(key=lambda x: x.html_url or '', reverse=True)
        
        # Limit to requested number
        recent_docs = minute_docs[:limit]
        
        # Scrape content for each document
        scraped_docs = []
        for doc in recent_docs:
            scraped_doc = self.scrape_document_content(doc)
            scraped_docs.append(scraped_doc)
            
            # Additional respectful delay between document scrapes
            time.sleep(self.request_delay)
        
        return scraped_docs
    
    def save_data(self, documents: List[FOMCDocument], filename: str = None):
        """Save scraped documents to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'fomc_minutes_{timestamp}.json'
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Convert documents to dictionaries for JSON serialization
        data = {
            'scraped_at': datetime.now().isoformat(),
            'total_documents': len(documents),
            'documents': [asdict(doc) for doc in documents]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Data saved to: {filepath}")
        return filepath
    
    def run_complete_pipeline(self, limit: int = 3) -> Dict:
        """
        Run the complete scraping, validation, and transformation pipeline
        Returns comprehensive results
        """
        self.logger.info("Starting complete FED scraping pipeline")
        
        try:
            # Step 1: Scrape documents
            documents = self.scrape_recent_minutes(limit)
            
            # Step 2: Transform and analyze
            transformed_data = self.transformer.process_documents(documents)
            
            # Step 3: Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save raw data
            raw_file = self.save_data(documents, f'raw_fomc_data_{timestamp}.json')
            
            # Save analysis
            analysis_file = os.path.join(self.data_dir, f'fomc_analysis_{timestamp}.json')
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(transformed_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Generate summary
            summary = self._generate_summary(documents, transformed_data)
            summary_file = os.path.join(self.data_dir, f'executive_summary_{timestamp}.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            results = {
                'success': True,
                'documents_processed': len(documents),
                'files_created': {
                    'raw_data': raw_file,
                    'analysis': analysis_file,
                    'summary': summary_file
                },
                'summary': summary,
                'transformed_data': transformed_data
            }
            
            self.logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'documents_processed': 0
            }
    
    def _generate_summary(self, documents: List[FOMCDocument], transformed_data: Dict) -> str:
        """Generate executive summary"""
        summary_parts = []
        
        summary_parts.append("=== FEDERAL RESERVE FOMC MINUTES ANALYSIS ===")
        summary_parts.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        summary_parts.append("")
        
        # Data summary
        valid_docs = sum(1 for doc in documents if doc.validation_status == 'VALID')
        summary_parts.append(f"DOCUMENTS PROCESSED: {len(documents)}")
        summary_parts.append(f"VALIDATION SUCCESS: {valid_docs}/{len(documents)}")
        summary_parts.append("")
        
        # Market summary
        if transformed_data.get('market_summary'):
            market = transformed_data['market_summary']
            summary_parts.append("MARKET ANALYSIS:")
            summary_parts.append(f"• Policy Stance: {market.get('current_policy_stance', 'N/A').upper()}")
            summary_parts.append(f"• Sentiment Score: {market.get('current_sentiment_score', 0):.2f}")
            summary_parts.append(f"• Trend: {market.get('policy_trajectory', 'N/A').upper()}")
            if market.get('recommendation'):
                summary_parts.append(f"• Recommendation: {market['recommendation']}")
            summary_parts.append("")
        
        # Document details
        summary_parts.append("DOCUMENT DETAILS:")
        for i, doc in enumerate(documents, 1):
            summary_parts.append(f"{i}. {doc.meeting_date or 'Unknown Date'}")
            summary_parts.append(f"   Status: {doc.validation_status or 'Not Validated'}")
            summary_parts.append(f"   Content: {len(doc.content) if doc.content else 0} characters")
        
        summary_parts.append("")
        summary_parts.append("=== END ANALYSIS ===")
        
        return "\n".join(summary_parts)

def main():
    """Main execution function"""
    print("🏦 Federal Reserve FOMC Minutes Scraper")
    print("=" * 50)
    
    scraper = FEDScraper()
    
    # Run complete pipeline
    results = scraper.run_complete_pipeline(limit=4)
    
    if results['success']:
        print(f"✅ Success! Processed {results['documents_processed']} documents")
        print(f"\n📁 Files created:")
        for file_type, filepath in results['files_created'].items():
            print(f"   - {file_type}: {filepath}")
        
        print(f"\n📋 Executive Summary:")
        print("-" * 40)
        print(results['summary'])
        
    else:
        print(f"❌ Pipeline failed: {results['error']}")

if __name__ == "__main__":
    main()