import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Structure for validation results"""
    status: str  # 'VALID', 'WARNING', 'INVALID'
    issues: List[str]
    score: float  # 0.0 to 1.0
    metadata: Dict[str, Any]

class FEDDataValidator:
    """
    Comprehensive data validation for Federal Reserve FOMC documents
    Implements quality assurance rules based on expected document structure
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Expected FOMC participants (updated as of 2025)
        self.expected_fomc_members = {
            'Jerome H. Powell',  # Chair
            'John C. Williams',  # Vice Chair, New York Fed President
            'Michael S. Barr',
            'Michelle W. Bowman',
            'Lisa D. Cook',
            'Austan D. Goolsbee',  # Chicago Fed President
            'Philip N. Jefferson',
            'Adriana D. Kugler',
            'Christopher J. Waller'
        }
        
        # Expected section headers in FOMC minutes
        self.required_sections = [
            'developments in financial markets',
            'economic outlook',
            'monetary policy',
            'participants',
            'committee policy action'
        ]
        
        # Content length expectations (in characters)
        self.min_content_length = 3000  # Reduced for more flexible validation
        self.max_content_length = 50000
        self.typical_length_range = (5000, 25000)
        
        # Date validation patterns
        self.date_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:[-–]\d{1,2})?,?\s+\d{4}\b',
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]
    
    def validate_document(self, document) -> Dict[str, Any]:
        """
        Master validation function for FOMC documents
        Returns comprehensive validation results
        """
        self.logger.info(f"Validating document: {document.meeting_date or 'Unknown'}")
        
        issues = []
        score = 1.0
        metadata = {}
        
        # Run all validation checks
        validations = [
            self._validate_content_structure(document),
            self._validate_dates(document),
            self._validate_participants(document),
            self._validate_policy_content(document),
            self._validate_technical_quality(document)
        ]
        
        # Aggregate results
        for validation in validations:
            issues.extend(validation.issues)
            score *= validation.score
            metadata.update(validation.metadata)
        
        # Determine overall status
        if score >= 0.7:  # Lowered threshold for more lenient validation
            status = 'VALID'
        elif score >= 0.4:
            status = 'WARNING'
        else:
            status = 'INVALID'
        
        result = {
            'status': status,
            'issues': issues,
            'score': round(score, 3),
            'metadata': metadata,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Validation complete. Status: {status}, Score: {score:.3f}")
        return result
    
    def _validate_content_structure(self, document) -> ValidationResult:
        """Validate the overall structure and completeness of document content"""
        issues = []
        score = 1.0
        metadata = {}
        
        if not document.content:
            issues.append("No content found in document")
            return ValidationResult('INVALID', issues, 0.0, metadata)
        
        content = document.content.lower()
        content_length = len(document.content)
        
        # Check content length
        metadata['content_length'] = content_length
        if content_length < self.min_content_length:
            issues.append(f"Content short ({content_length} chars)")
            score *= 0.7  # Less harsh penalty
        elif content_length > self.max_content_length:
            issues.append(f"Content very long ({content_length} chars)")
            score *= 0.9
        
        # Check for required sections (more flexible matching)
        found_sections = []
        for section in self.required_sections:
            section_words = section.split()
            # Check if at least half the words from each section are present
            word_matches = sum(1 for word in section_words if word.lower() in content)
            if word_matches >= len(section_words) / 2:
                found_sections.append(section)
        
        metadata['found_sections'] = found_sections
        metadata['missing_sections'] = list(set(self.required_sections) - set(found_sections))
        
        section_coverage = len(found_sections) / len(self.required_sections)
        if section_coverage < 0.4:  # More lenient threshold
            issues.append(f"Missing key sections ({len(found_sections)}/{len(self.required_sections)})")
            score *= max(0.5, section_coverage)
        elif section_coverage < 0.6:
            issues.append(f"Some sections missing ({len(found_sections)}/{len(self.required_sections)})")
            score *= 0.9
        
        # Check for HTML artifacts (less strict)
        html_artifacts = re.findall(r'<[^>]+>', document.content)
        if len(html_artifacts) > 10:  # Only penalize if many artifacts
            issues.append(f"HTML parsing issues detected")
            score *= 0.8
            metadata['html_artifacts_count'] = len(html_artifacts)
        
        # Check for reasonable paragraph structure
        paragraphs = [p.strip() for p in document.content.split('\n') if p.strip()]
        metadata['paragraph_count'] = len(paragraphs)
        
        if len(paragraphs) < 5:
            issues.append("Limited paragraph structure")
            score *= 0.8
        
        return ValidationResult('VALID' if score > 0.7 else 'WARNING', issues, score, metadata)
    
    def _validate_dates(self, document) -> ValidationResult:
        """Validate date information in the document"""
        issues = []
        score = 1.0
        metadata = {}
        
        # Validate meeting date format
        if document.meeting_date:
            try:
                parsed_date = self._parse_flexible_date(document.meeting_date)
                if parsed_date:
                    metadata['parsed_meeting_date'] = parsed_date.isoformat()
                    
                    # Check if date is reasonable
                    now = datetime.now()
                    if parsed_date > now + timedelta(days=365):
                        issues.append("Meeting date far in future")
                        score *= 0.8
                    elif parsed_date < datetime(2008, 1, 1):
                        issues.append("Meeting date unusually old")
                        score *= 0.9
                else:
                    issues.append(f"Could not parse meeting date: {document.meeting_date}")
                    score *= 0.9  # Less harsh penalty
            except Exception as e:
                issues.append(f"Date parsing error")
                score *= 0.9
        else:
            issues.append("No meeting date provided")
            score *= 0.8  # Less harsh for missing date
        
        # Look for dates in content (more lenient)
        if document.content:
            date_matches = []
            for pattern in self.date_patterns:
                matches = re.findall(pattern, document.content, re.IGNORECASE)
                date_matches.extend(matches)
            
            metadata['dates_in_content'] = len(date_matches)
            if len(date_matches) < 2:  # Reduced expectation
                issues.append("Few date references in content")
                score *= 0.95
        
        return ValidationResult('VALID' if score > 0.8 else 'WARNING', issues, score, metadata)
    
    def _validate_participants(self, document) -> ValidationResult:
        """Validate FOMC participants mentioned in the document"""
        issues = []
        score = 1.0
        metadata = {}
        
        if not document.content:
            return ValidationResult('WARNING', ['No content to validate participants'], 0.5, metadata)
        
        content_lower = document.content.lower()
        found_members = []
        
        # Check for expected FOMC members (more flexible matching)
        for member in self.expected_fomc_members:
            full_name_lower = member.lower()
            last_name = member.split()[-1].lower()
            first_name = member.split()[0].lower()
            
            if (full_name_lower in content_lower or 
                last_name in content_lower or 
                first_name in content_lower):
                found_members.append(member)
        
        metadata['found_fomc_members'] = found_members
        metadata['found_members_count'] = len(found_members)
        
        # More lenient expectations for member mentions
        if len(found_members) == 0:
            issues.append("No FOMC members found")
            score *= 0.7  # Less harsh
        elif len(found_members) < 2:
            issues.append(f"Few FOMC members mentioned ({len(found_members)})")
            score *= 0.9
        
        # Look for general participant indicators
        participant_indicators = [
            'participants', 'members', 'committee', 'governors',
            'presidents', 'policymakers', 'officials', 'chair', 'vice chair'
        ]
        
        found_indicators = [ind for ind in participant_indicators if ind in content_lower]
        metadata['participant_indicators'] = found_indicators
        
        if len(found_indicators) < 2:  # Reduced threshold
            issues.append("Limited participant terminology")
            score *= 0.95
        
        return ValidationResult('VALID' if score > 0.8 else 'WARNING', issues, score, metadata)
    
    def _validate_policy_content(self, document) -> ValidationResult:
        """Validate monetary policy content and decisions"""
        issues = []
        score = 1.0
        metadata = {}
        
        if not document.content:
            return ValidationResult('WARNING', ['No content to validate policy'], 0.5, metadata)
        
        content_lower = document.content.lower()
        
        # Look for key monetary policy terms (expanded list)
        policy_terms = [
            'federal funds rate', 'interest rate', 'target range', 'basis points',
            'monetary policy', 'inflation', 'employment', 'unemployment',
            'economic outlook', 'balance sheet', 'asset purchases',
            'quantitative easing', 'tightening', 'accommodation', 'gdp',
            'labor market', 'price stability', 'dual mandate'
        ]
        
        found_terms = [term for term in policy_terms if term in content_lower]
        metadata['policy_terms_found'] = found_terms
        metadata['policy_terms_count'] = len(found_terms)
        
        if len(found_terms) < 3:  # Reduced threshold
            issues.append(f"Limited monetary policy terminology ({len(found_terms)} terms)")
            score *= 0.8
        
        # Look for voting patterns (more flexible)
        voting_indicators = ['voted', 'dissent', 'unanimous', 'favor', 'against', 'decision', 'agreed']
        found_voting = [term for term in voting_indicators if term in content_lower]
        metadata['voting_indicators'] = found_voting
        
        if not found_voting:
            issues.append("No voting information found")
            score *= 0.9  # Less harsh
        
        # Check for economic data references
        economic_indicators = [
            'gdp', 'unemployment', 'inflation', 'cpi', 'pce', 'employment',
            'labor market', 'economic growth', 'recession', 'expansion',
            'financial conditions', 'markets'
        ]
        
        found_economic = [term for term in economic_indicators if term in content_lower]
        metadata['economic_indicators'] = found_economic
        
        if len(found_economic) < 2:  # Reduced threshold
            issues.append("Limited economic indicators")
            score *= 0.95
        
        return ValidationResult('VALID' if score > 0.8 else 'WARNING', issues, score, metadata)
    
    def _validate_technical_quality(self, document) -> ValidationResult:
        """Validate technical aspects of the scraped content"""
        issues = []
        score = 1.0
        metadata = {}
        
        if not document.content:
            return ValidationResult('INVALID', ['No content'], 0.0, metadata)
        
        # Check character encoding issues
        encoding_issues = []
        problematic_chars = ['�', '\\x', '\x00']
        for char in problematic_chars:
            if char in document.content:
                encoding_issues.append(char)
        
        if encoding_issues:
            issues.append(f"Encoding issues detected")
            score *= 0.9
            metadata['encoding_issues'] = encoding_issues
        
        # Check for excessive whitespace (parsing issues)
        lines = document.content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        total_lines = len(lines)
        
        if total_lines > 0:
            content_line_ratio = len(non_empty_lines) / total_lines
            metadata['content_line_ratio'] = round(content_line_ratio, 3)
            
            if content_line_ratio < 0.3:  # More lenient
                issues.append(f"High ratio of empty lines")
                score *= 0.9
        
        # Check for reasonable word count
        words = document.content.split()
        word_count = len(words)
        metadata['word_count'] = word_count
        
        if word_count < 500:  # Reduced threshold
            issues.append(f"Low word count ({word_count})")
            score *= 0.7
        elif word_count > 20000:
            issues.append(f"Very high word count ({word_count})")
            score *= 0.95
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', document.content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            metadata['avg_sentence_length'] = round(avg_sentence_length, 2)
            metadata['sentence_count'] = len(sentences)
            
            if avg_sentence_length < 3:
                issues.append(f"Very short sentences (avg: {avg_sentence_length:.1f} words)")
                score *= 0.9
            elif avg_sentence_length > 60:
                issues.append(f"Very long sentences (avg: {avg_sentence_length:.1f} words)")
                score *= 0.95
        
        # Check for scraped timestamp
        if document.scraped_at:
            try:
                scraped_time = datetime.fromisoformat(document.scraped_at.replace('Z', '+00:00'))
                metadata['scraped_timestamp'] = scraped_time.isoformat()
            except Exception:
                issues.append("Invalid scraped timestamp")
                score *= 0.99
        
        return ValidationResult('VALID' if score > 0.8 else 'WARNING', issues, score, metadata)
    
    def _parse_flexible_date(self, date_str: str) -> Optional[datetime]:
        """Parse dates in various formats commonly used by the Fed"""
        if not date_str:
            return None
        
        # Common Fed date formats
        date_formats = [
            '%B %d, %Y',           # January 28, 2025
            '%B %d-%d, %Y',        # January 28-29, 2025  
            '%B %d–%d, %Y',        # January 28–29, 2025 (en dash)
            '%m/%d/%Y',            # 01/28/2025
            '%Y-%m-%d',            # 2025-01-28
            '%d %B %Y',            # 28 January 2025
            '%B %Y'                # January 2025
        ]
        
        # Clean the date string
        date_str = date_str.strip()
        
        # Handle date ranges - take the first date
        range_patterns = [
            r'(\w+ \d{1,2})[-–](\d{1,2}, \d{4})',  # January 28-29, 2025
            r'(\w+ \d{1,2})[-–](\d{1,2}), (\d{4})'  # January 28-29, 2025
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, date_str)
            if match:
                if len(match.groups()) == 2:
                    month_day = match.group(1)
                    day_year = match.group(2)
                    date_str = f"{month_day}, {day_year}"
                elif len(match.groups()) == 3:
                    month_day = match.group(1)
                    year = match.group(3)
                    date_str = f"{month_day}, {year}"
                break
        
        # Try each format
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try partial matching for month names
        month_match = re.search(r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', date_str)
        if month_match:
            try:
                return datetime.strptime(f"{month_match.group(1)} {month_match.group(2)}, {month_match.group(3)}", '%B %d, %Y')
            except ValueError:
                pass
        
        return None
    
    def validate_batch(self, documents: List) -> Dict[str, Any]:
        """Validate a batch of documents and provide summary statistics"""
        self.logger.info(f"Validating batch of {len(documents)} documents")
        
        results = []
        for doc in documents:
            result = self.validate_document(doc)
            results.append(result)
        
        # Calculate batch statistics
        valid_count = sum(1 for r in results if r['status'] == 'VALID')
        warning_count = sum(1 for r in results if r['status'] == 'WARNING')
        invalid_count = sum(1 for r in results if r['status'] == 'INVALID')
        
        avg_score = sum(r['score'] for r in results) / len(results) if results else 0
        
        # Collect common issues
        all_issues = []
        for r in results:
            all_issues.extend(r['issues'])
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        batch_summary = {
            'total_documents': len(documents),
            'valid_documents': valid_count,
            'warning_documents': warning_count,
            'invalid_documents': invalid_count,
            'average_score': round(avg_score, 3),
            'common_issues': common_issues,
            'individual_results': results,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Batch validation complete: {valid_count} valid, {warning_count} warnings, {invalid_count} invalid")
        
        return batch_summary

def main():
    """Test the validator with sample data"""
    # Simple test data structure
    class TestDoc:
        def __init__(self, content, meeting_date=None, scraped_at=None):
            self.content = content
            self.meeting_date = meeting_date
            self.scraped_at = scraped_at
            self.release_date = None
            self.document_type = "minutes"
    
    validator = FEDDataValidator()
    
    # Create test document
    test_doc = TestDoc(
        content="""
        A joint meeting of the Federal Open Market Committee and the Board of Governors 
        was held on January 28-29, 2025. The Committee discussed developments in financial 
        markets and the economic outlook. Participants noted inflation trends and employment 
        conditions. Jerome H. Powell chaired the meeting. The Committee voted unanimously 
        to maintain the federal funds rate target range. John C. Williams and other 
        participants discussed monetary policy considerations. The unemployment rate remained 
        stable while inflation showed signs of moderating.
        """,
        meeting_date="January 28-29, 2025",
        scraped_at=datetime.now().isoformat()
    )
    
    # Validate the test document
    result = validator.validate_document(test_doc)
    
    print("=== FED DATA VALIDATOR TEST ===")
    print(f"Status: {result['status']}")
    print(f"Score: {result['score']}")
    print(f"Issues: {result['issues']}")
    print("\nValidator test completed successfully!")
    
    return result

if __name__ == "__main__":
    main()