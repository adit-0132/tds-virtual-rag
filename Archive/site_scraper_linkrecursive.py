from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import time

class TDSScraper:
    def __init__(self):
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Initialize the driver
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        
    def scrape_course_content(self, base_url="https://tds.s-anand.net/#/2025-01/"):
        try:
            print(f"Loading {base_url}")
            self.driver.get(base_url)
            
            # Wait for content to load
            time.sleep(3)
            
            # Get the page source after JavaScript execution
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract main content
            content_data = {
                "course_title": "",
                "sections": [],
                "links": []
            }
            
            # Try to find the main content area
            # This will depend on the actual HTML structure
            main_content = soup.find('main') or soup.find('div', class_='content') or soup.find('body')
            
            if main_content:
                # Extract title
                title = main_content.find('h1') or main_content.find('title')
                if title:
                    content_data["course_title"] = title.get_text().strip()
                
                # Extract all headings and their content
                headings = main_content.find_all(['h1', 'h2', 'h3', 'h4'])
                
                for heading in headings:
                    section = {
                        "heading": heading.get_text().strip(),
                        "level": heading.name,
                        "content": []
                    }
                    
                    # Get content after this heading until next heading
                    current = heading.next_sibling
                    while current and current.name not in ['h1', 'h2', 'h3', 'h4']:
                        if hasattr(current, 'get_text'):
                            text = current.get_text().strip()
                            if text:
                                section["content"].append(text)
                        current = current.next_sibling
                    
                    content_data["sections"].append(section)
                
                # Extract all links for potential navigation
                links = main_content.find_all('a', href=True)
                for link in links:
                    if link['href'].startswith('#') or 'tds.s-anand.net' in link['href']:
                        content_data["links"].append({
                            "text": link.get_text().strip(),
                            "href": link['href']
                        })
            
            return content_data
            
        except Exception as e:
            print(f"Error scraping: {e}")
            return None
    
    def explore_navigation(self):
        """Try to find all course sections/pages"""
        try:
            # Look for navigation elements
            nav_elements = self.driver.find_elements(By.TAG_NAME, "nav")
            nav_elements += self.driver.find_elements(By.CLASS_NAME, "nav")
            nav_elements += self.driver.find_elements(By.CLASS_NAME, "menu")
            
            sections = []
            for nav in nav_elements:
                links = nav.find_elements(By.TAG_NAME, "a")
                for link in links:
                    href = link.get_attribute("href")
                    text = link.text.strip()
                    if href and text:
                        sections.append({"text": text, "href": href})
            
            return sections
            
        except Exception as e:
            print(f"Error exploring navigation: {e}")
            return []
    
    def get_all_links(self):
        """Get all internal links from the main page"""
        try:
            # Get all links
            links = self.driver.find_elements(By.TAG_NAME, "a")
            
            unique_links = set()
            for link in links:
                href = link.get_attribute("href")
                text = link.text.strip()
                
                if href and (href.startswith("https://tds.s-anand.net") or href.startswith("#")):
                    # Convert relative links to full URLs
                    if href.startswith("#"):
                        href = f"https://tds.s-anand.net/{href}"
                    
                    unique_links.add((href, text))
            
            return list(unique_links)
            
        except Exception as e:
            print(f"Error getting links: {e}")
            return []
    
    def scrape_all_sections(self):
        """Scrape the main page and all linked pages"""
        all_content = []
        visited_urls = set()
        
        # Start with main page
        base_url = "https://tds.s-anand.net/#/2025-01/"
        
        # Scrape main page
        print(f"Scraping main page: {base_url}")
        main_content = self.scrape_course_content(base_url)
        if main_content:
            main_content["url"] = base_url
            all_content.append(main_content)
            visited_urls.add(base_url)
        
        # Get all links from main page
        all_links = self.get_all_links()
        print(f"Found {len(all_links)} links to explore")
        
        # Visit each unique link
        for href, link_text in all_links:
            if href not in visited_urls:
                try:
                    print(f"Scraping: {link_text} ({href})")
                    
                    # Navigate to the link
                    self.driver.get(href)
                    time.sleep(3)  # Wait for content to load
                    
                    # Scrape content from this page
                    page_content = self.scrape_course_content(href)
                    
                    if page_content and page_content.get("sections"):
                        page_content["url"] = href
                        page_content["link_text"] = link_text
                        all_content.append(page_content)
                        
                    visited_urls.add(href)
                    
                except Exception as e:
                    print(f"Error scraping {href}: {e}")
                    continue
        
        print(f"Successfully scraped {len(all_content)} pages")
        return all_content
    
    def save_to_json(self, data, filename="tds_course_content_links.json"):
        """Save scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Content saved to {filename}")
    
    def close(self):
        """Close the browser"""
        self.driver.quit()

# Usage example
if __name__ == "__main__":
    scraper = TDSScraper()
    
    try:
        # Scrape all content
        content = scraper.scrape_all_sections()
        
        # Save to file
        scraper.save_to_json(content)
        
        print(f"Scraped {len(content)} sections")
        for section in content:
            print(f"- {section.get('course_title', 'Unknown')} ({len(section.get('sections', []))} subsections)")
            
    finally:
        scraper.close()