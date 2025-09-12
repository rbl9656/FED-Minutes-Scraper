"""
Federal Reserve Data Transformation Pipeline
Implements business logic and value-added analysis for FOMC minutes
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import Counter, defaultdict
import json
import os
from dataclasses import dataclass, asdict
import statistics

@dataclass
class PolicySignal:
    """Structure for monetary policy signals detected in text"""
    signal_type: str  # 'hawkish', 'dovish', 'neutral'
    confidence: float  # 0.0 to 1.0
    evidence: List[str]  # Supporting text snippets
    context: str  # Where in document this was found

@dataclass
class EconomicIndicator:
    """Structure for economic indicators mentioned in minutes"""
    indicator: str
    mentions: int
    context_snippets: List[str]
    sentiment: str  # 'positive', 'negative', 'neutral'

@dataclass
class AnalysisResult:
    """Comprehensive analysis result for FOMC documents"""
    document_id: str
    meeting_date: str
    policy_stance: str
    key_themes: List[str]
    policy_signals: List[PolicySignal]
    economic_indicators: List[EconomicIndicator]
    voting_analysis: Dict[str, Any]
    market_implications: List[str]
    sentiment_score: float
    analysis_timestamp: str

class FEDDataTransformer:
    """
    Advanced data transformation and analysis for Federal Reserve content
    Extracts business insights and market-relevant intelligence
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Hawkish (tightening) policy indicators
        self.hawkish_terms = {
            'strong': ['aggressive', 'accelerated', 'brisk', 'robust', 'vigorous', 'strong'],
            'tightening': ['tighten', 'restrictive', 'contractionary', 'reduce accommodation', 'less accommodative'],
            'inflation_concern': ['upside risks', 'inflationary pressures', 'price pressures', 'elevated inflation'],
            'rate_hikes': ['increase', 'raise rates', 'higher rates', 'more aggressive', 'tightening cycle'],
            'economic_strength': ['strong growth', 'tight labor market', 'overheating', 'above trend']
        }
        
        # Dovish (easing) policy indicators  
        self.dovish_terms = {
            'weakness': ['slowdown', 'weakening', 'softening', 'declining', 'deteriorating'],
            'easing': ['accommodative', 'supportive', 'stimulus', 'lower rates', 'easier policy'],
            'growth_concern': ['downside risks', 'economic uncertainty', 'slowing growth', 'headwinds'],
            'employment_concern': ['job losses', 'unemployment rising', 'labor market slack', 'layoffs'],
            'cautious': ['patience', 'gradual', 'measured', 'data dependent', 'wait and see']
        }
        
        # Key economic indicators to track
        self.economic_indicators = {
            'inflation': ['inflation', 'cpi', 'pce', 'price level', 'deflationary', 'disinflation'],
            'employment': ['employment', 'unemployment', 'jobs', 'labor market', 'payroll', 'jobless'],
            'growth': ['gdp', 'economic growth', 'expansion', 'recession', 'recovery', 'output'],
            'financial_conditions': ['financial conditions', 'credit', 'liquidity', 'markets', 'lending'],
            'housing': ['housing', 'real estate', 'mortgage', 'home prices', 'construction'],
            'international': ['global', 'trade', 'tariffs', 'geopolitical', 'foreign', 'international']
        }
        
        # Voting pattern indicators
        self.voting_patterns = [
            r'voted unanimously',
            r'(\d+)\s+voted\s+in\s+favor',
            r'(\d+)\s+voted\s+against', 
            r'dissented?',
            r'(\w+(?:\s+\w+)*)\s+dissented',
            r'(\w+(?:\s+\w+)*)\s+voted\s+against'
        ]
    
    def process_documents(self, documents: List) -> Dict[str, Any]:
        """
        Main processing pipeline for FOMC documents
        Returns comprehensive analysis with market insights
        """
        self.logger.info(f"Processing {len(documents)} documents for analysis")
        
        analysis_results = []
        aggregate_insights = {
            'policy_evolution': {},
            'key_themes_trends': {},
            'sentiment_trend': [],
            'voting_patterns': {},
            'market_signals': []
        }
        
        for i, document in enumerate(documents):
            if not document.content:
                self.logger.warning(f"Skipping document {i+1}: No content")
                continue
            
            # Perform comprehensive analysis
            analysis = self.analyze_document(document, document_id=f"doc_{i+1}")
            analysis_results.append(analysis)
            
            # Update aggregate insights
            self._update_aggregate_insights(analysis, aggregate_insights)
        
        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(analysis_results)
        
        # Create final transformed dataset
        transformed_data = {
            'analysis_metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'documents_processed': len(documents),
                'successful_analyses': len(analysis_results)
            },
            'individual_analyses': [asdict(analysis) for analysis in analysis_results],
            'aggregate_insights': aggregate_insights,
            'comparative_analysis': comparative_analysis,
            'market_summary': self._generate_market_summary(analysis_results)
        }
        
        self.logger.info("Document processing completed successfully")
        return transformed_data
    
    def analyze_document(self, document, document_id: str) -> AnalysisResult:
        """
        Comprehensive analysis of a single FOMC document
        Extracts policy signals, themes, and market implications
        """
        content = document.content.lower() if document.content else ""
        
        # Extract policy signals
        policy_signals = self._extract_policy_signals(content)
        
        # Determine overall policy stance
        policy_stance = self._determine_policy_stance(policy_signals)
        
        # Extract key themes
        key_themes = self._extract_key_themes(content)
        
        # Analyze economic indicators
        economic_indicators = self._analyze_economic_indicators(content)
        
        # Analyze voting patterns
        voting_analysis = self._analyze_voting_patterns(document.content if document.content else "")
        
        # Generate market implications
        market_implications = self._generate_market_implications(policy_signals, economic_indicators)
        
        # Calculate sentiment score
        sentiment_score = self._calculate_sentiment_score(policy_signals, economic_indicators)
        
        return AnalysisResult(
            document_id=document_id,
            meeting_date=document.meeting_date or "Unknown",
            policy_stance=policy_stance,
            key_themes=key_themes,
            policy_signals=policy_signals,
            economic_indicators=economic_indicators,
            voting_analysis=voting_analysis,
            market_implications=market_implications,
            sentiment_score=sentiment_score,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _extract_policy_signals(self, content: str) -> List[PolicySignal]:
        """Extract and classify monetary policy signals from text"""
        signals = []
        
        # Analyze hawkish signals
        hawkish_evidence = []
        hawkish_confidence = 0.0
        
        for category, terms in self.hawkish_terms.items():
            for term in terms:
                if term.lower() in content:
                    # Find context around the term
                    contexts = self._find_term_contexts(content, term, context_window=50)
                    hawkish_evidence.extend([f"{category}: {ctx}" for ctx in contexts])
                    hawkish_confidence += 0.1 * len(contexts)
        
        if hawkish_evidence:
            signals.append(PolicySignal(
                signal_type='hawkish',
                confidence=min(hawkish_confidence, 1.0),
                evidence=hawkish_evidence[:5],  # Limit to top 5 pieces of evidence
                context='policy_stance_analysis'
            ))
        
        # Analyze dovish signals
        dovish_evidence = []
        dovish_confidence = 0.0
        
        for category, terms in self.dovish_terms.items():
            for term in terms:
                if term.lower() in content:
                    contexts = self._find_term_contexts(content, term, context_window=50)
                    dovish_evidence.extend([f"{category}: {ctx}" for ctx in contexts])
                    dovish_confidence += 0.1 * len(contexts)
        
        if dovish_evidence:
            signals.append(PolicySignal(
                signal_type='dovish',
                confidence=min(dovish_confidence, 1.0),
                evidence=dovish_evidence[:5],
                context='policy_stance_analysis'
            ))
        
        # Look for neutral/balanced language
        neutral_terms = ['balance', 'measured', 'appropriate', 'monitor', 'assess', 'evaluate']
        neutral_evidence = []
        
        for term in neutral_terms:
            if term in content:
                contexts = self._find_term_contexts(content, term, context_window=30)
                neutral_evidence.extend(contexts)
        
        if neutral_evidence:
            signals.append(PolicySignal(
                signal_type='neutral',
                confidence=min(len(neutral_evidence) * 0.15, 1.0),
                evidence=neutral_evidence[:3],
                context='balanced_policy_language'
            ))
        
        return signals
    
    def _find_term_contexts(self, content: str, term: str, context_window: int = 50) -> List[str]:
        """Find contextual snippets around specific terms"""
        contexts = []
        term_lower = term.lower()
        
        # Find all occurrences
        start = 0
        while True:
            pos = content.find(term_lower, start)
            if pos == -1:
                break
            
            # Extract context around the term
            context_start = max(0, pos - context_window)
            context_end = min(len(content), pos + len(term_lower) + context_window)
            context = content[context_start:context_end].strip()
            
            if context and len(context) > len(term_lower):
                contexts.append(context)
            
            start = pos + 1
        
        return contexts[:3]  # Limit to 3 contexts per term
    
    def _determine_policy_stance(self, policy_signals: List[PolicySignal]) -> str:
        """Determine overall policy stance from signals"""
        if not policy_signals:
            return "neutral"
        
        # Weight signals by confidence
        hawkish_weight = sum(sig.confidence for sig in policy_signals if sig.signal_type == 'hawkish')
        dovish_weight = sum(sig.confidence for sig in policy_signals if sig.signal_type == 'dovish')
        neutral_weight = sum(sig.confidence for sig in policy_signals if sig.signal_type == 'neutral')
        
        # Determine stance based on weighted signals
        if hawkish_weight > dovish_weight * 1.2:
            return "hawkish"
        elif dovish_weight > hawkish_weight * 1.2:
            return "dovish"
        else:
            return "neutral"
    
    def _extract_key_themes(self, content: str) -> List[str]:
        """Extract key thematic content from the document"""
        themes = []
        
        # Define theme keywords
        theme_keywords = {
            'inflation_outlook': ['inflation expectations', 'price stability', 'inflation target', '2 percent'],
            'employment_conditions': ['maximum employment', 'labor market conditions', 'unemployment rate', 'job growth'],
            'financial_stability': ['financial stability', 'systemic risk', 'financial conditions', 'credit conditions'],
            'international_factors': ['global economy', 'international developments', 'trade tensions', 'geopolitical'],
            'forward_guidance': ['forward guidance', 'future policy', 'policy path', 'communication'],
            'balance_sheet': ['balance sheet', 'asset purchases', 'quantitative easing', 'securities holdings'],
            'economic_uncertainty': ['uncertainty', 'risks to outlook', 'downside risks', 'upside risks']
        }
        
        for theme, keywords in theme_keywords.items():
            mentions = sum(1 for keyword in keywords if keyword in content)
            if mentions > 0:
                themes.append(f"{theme.replace('_', ' ').title()} ({mentions} mentions)")
        
        # Extract frequent important phrases (3-4 word phrases mentioned multiple times)
        phrases = re.findall(r'\b(?:\w+\s+){2,3}\w+\b', content)
        phrase_counter = Counter(phrases)
        
        # Filter for economically relevant phrases
        relevant_phrases = []
        economic_keywords = ['rate', 'market', 'economic', 'policy', 'inflation', 'employment', 'growth']
        
        for phrase, count in phrase_counter.most_common(20):
            if count >= 2 and any(keyword in phrase.lower() for keyword in economic_keywords):
                relevant_phrases.append(f"'{phrase}' ({count}x)")
        
        themes.extend(relevant_phrases[:5])  # Add top 5 relevant phrases
        
        return themes
    
    def _analyze_economic_indicators(self, content: str) -> List[EconomicIndicator]:
        """Analyze mentions and sentiment around economic indicators"""
        indicators = []
        
        for indicator_name, keywords in self.economic_indicators.items():
            total_mentions = 0
            context_snippets = []
            
            for keyword in keywords:
                if keyword in content:
                    mentions = content.count(keyword)
                    total_mentions += mentions
                    
                    # Get context snippets
                    contexts = self._find_term_contexts(content, keyword, context_window=40)
                    context_snippets.extend(contexts)
            
            if total_mentions > 0:
                # Determine sentiment based on surrounding words
                sentiment = self._determine_indicator_sentiment(context_snippets)
                
                indicators.append(EconomicIndicator(
                    indicator=indicator_name.replace('_', ' ').title(),
                    mentions=total_mentions,
                    context_snippets=context_snippets[:3],  # Top 3 contexts
                    sentiment=sentiment
                ))
        
        # Sort by mention frequency
        indicators.sort(key=lambda x: x.mentions, reverse=True)
        
        return indicators
    
    def _determine_indicator_sentiment(self, contexts: List[str]) -> str:
        """Determine sentiment around economic indicator mentions"""
        if not contexts:
            return "neutral"
        
        positive_words = ['strong', 'robust', 'improving', 'growth', 'increase', 'rising', 'stable', 'solid', 'healthy']
        negative_words = ['weak', 'declining', 'falling', 'concerns', 'risks', 'uncertainty', 'slow', 'deteriorating']
        
        all_text = ' '.join(contexts).lower()
        
        positive_count = sum(1 for word in positive_words if word in all_text)
        negative_count = sum(1 for word in negative_words if word in all_text)
        
        if positive_count > negative_count * 1.2:
            return "positive"
        elif negative_count > positive_count * 1.2:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_voting_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze voting patterns and dissents from the minutes"""
        voting_info = {
            'unanimous': False,
            'dissents': [],
            'voting_details': {},
            'consensus_strength': 'unknown'
        }
        
        if not content:
            return voting_info
        
        content_lower = content.lower()
        
        # Check for unanimous voting
        unanimous_phrases = [
            'voted unanimously', 'unanimous vote', 'unanimously approved',
            'all members voted', 'consensus decision'
        ]
        
        if any(phrase in content_lower for phrase in unanimous_phrases):
            voting_info['unanimous'] = True
            voting_info['consensus_strength'] = 'strong'
        
        # Look for dissents
        dissent_patterns = [
            r'(\w+(?:\s+\w+)*)\s+dissented',
            r'(\w+(?:\s+\w+)*)\s+voted against',
            r'dissent.*?(\w+(?:\s+\w+)*)',
        ]
        
        for pattern in dissent_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                dissenter = match.strip()
                if len(dissenter.split()) <= 3 and dissenter.lower() not in ['the', 'this', 'that', 'committee']:
                    voting_info['dissents'].append(dissenter)
        
        # Remove duplicates and common words
        voting_info['dissents'] = list(set([d for d in voting_info['dissents'] 
                                           if len(d.split()) >= 2]))
        
        # Determine consensus strength
        if not voting_info['unanimous']:
            if len(voting_info['dissents']) == 0:
                voting_info['consensus_strength'] = 'likely_strong'
            elif len(voting_info['dissents']) <= 2:
                voting_info['consensus_strength'] = 'moderate'
            else:
                voting_info['consensus_strength'] = 'weak'
        
        return voting_info
    
    def _generate_market_implications(self, policy_signals: List[PolicySignal], 
                                    economic_indicators: List[EconomicIndicator]) -> List[str]:
        """Generate market-relevant implications from the analysis"""
        implications = []
        
        # Policy stance implications
        hawkish_signals = [s for s in policy_signals if s.signal_type == 'hawkish']
        dovish_signals = [s for s in policy_signals if s.signal_type == 'dovish']
        
        if hawkish_signals:
            avg_confidence = statistics.mean([s.confidence for s in hawkish_signals])
            if avg_confidence > 0.5:
                implications.append("Hawkish signals suggest potential for higher interest rates")
                implications.append("Bond yields may face upward pressure")
                implications.append("USD strength likely to continue")
        
        if dovish_signals:
            avg_confidence = statistics.mean([s.confidence for s in dovish_signals])
            if avg_confidence > 0.5:
                implications.append("Dovish tone suggests accommodative policy stance")
                implications.append("Equity markets may benefit from lower rate expectations")
                implications.append("Bond prices could see support")
        
        # Economic indicator implications
        inflation_indicators = [ind for ind in economic_indicators if 'inflation' in ind.indicator.lower()]
        employment_indicators = [ind for ind in economic_indicators if 'employment' in ind.indicator.lower()]
        
        if inflation_indicators:
            inf_ind = inflation_indicators[0]
            if inf_ind.sentiment == 'positive' and inf_ind.mentions > 2:
                implications.append("Inflation concerns may drive more restrictive policy")
            elif inf_ind.sentiment == 'negative':
                implications.append("Disinflationary trends may provide policy flexibility")
        
        if employment_indicators:
            emp_ind = employment_indicators[0]
            if emp_ind.sentiment == 'positive':
                implications.append("Strong labor market supports continued policy normalization")
            elif emp_ind.sentiment == 'negative':
                implications.append("Labor market concerns may limit policy tightening")
        
        return implications
    
    def _calculate_sentiment_score(self, policy_signals: List[PolicySignal], 
                                 economic_indicators: List[EconomicIndicator]) -> float:
        """Calculate overall sentiment score (-1 to 1, where -1 is very dovish, 1 is very hawkish)"""
        score = 0.0
        
        # Policy signals contribution
        for signal in policy_signals:
            if signal.signal_type == 'hawkish':
                score += signal.confidence * 0.5
            elif signal.signal_type == 'dovish':
                score -= signal.confidence * 0.5
        
        # Economic indicators contribution
        for indicator in economic_indicators:
            weight = min(indicator.mentions / 10.0, 0.2)  # Max weight of 0.2
            
            if indicator.sentiment == 'positive':
                score += weight
            elif indicator.sentiment == 'negative':
                score -= weight
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, score))
    
    def _update_aggregate_insights(self, analysis: AnalysisResult, aggregate: Dict[str, Any]):
        """Update aggregate insights with individual analysis results"""
        # Policy evolution tracking
        if analysis.meeting_date not in aggregate['policy_evolution']:
            aggregate['policy_evolution'][analysis.meeting_date] = {
                'stance': analysis.policy_stance,
                'sentiment_score': analysis.sentiment_score
            }
        
        # Theme tracking
        for theme in analysis.key_themes:
            theme_key = theme.split('(')[0].strip()  # Remove count info
            if theme_key not in aggregate['key_themes_trends']:
                aggregate['key_themes_trends'][theme_key] = 0
            aggregate['key_themes_trends'][theme_key] += 1
        
        # Sentiment trend
        aggregate['sentiment_trend'].append({
            'date': analysis.meeting_date,
            'score': analysis.sentiment_score,
            'stance': analysis.policy_stance
        })
        
        # Market signals
        aggregate['market_signals'].extend(analysis.market_implications)
    
    def _generate_comparative_analysis(self, analyses: List[AnalysisResult]) -> Dict[str, Any]:
        """Generate comparative analysis across multiple documents"""
        if not analyses:
            return {}
        
        # Policy stance evolution
        stances = [analysis.policy_stance for analysis in analyses]
        stance_counts = Counter(stances)
        
        # Sentiment evolution
        sentiment_scores = [analysis.sentiment_score for analysis in analyses]
        avg_sentiment = statistics.mean(sentiment_scores) if sentiment_scores else 0
        
        # Most common themes
        all_themes = []
        for analysis in analyses:
            all_themes.extend(analysis.key_themes)
        
        theme_counts = Counter([theme.split('(')[0].strip() for theme in all_themes])
        
        return {
            'policy_stance_distribution': dict(stance_counts),
            'average_sentiment_score': round(avg_sentiment, 3),
            'sentiment_range': {
                'min': min(sentiment_scores) if sentiment_scores else 0,
                'max': max(sentiment_scores) if sentiment_scores else 0
            },
            'most_common_themes': theme_counts.most_common(10),
            'total_market_implications': len(set().union(*[analysis.market_implications for analysis in analyses])),
            'consensus_strength': self._assess_overall_consensus(analyses)
        }
    
    def _assess_overall_consensus(self, analyses: List[AnalysisResult]) -> str:
        """Assess overall consensus strength across meetings"""
        unanimous_meetings = sum(1 for analysis in analyses 
                               if analysis.voting_analysis.get('unanimous', False))
        
        total_meetings = len(analyses)
        
        if unanimous_meetings == total_meetings:
            return "very_strong"
        elif unanimous_meetings / total_meetings > 0.8:
            return "strong"
        elif unanimous_meetings / total_meetings > 0.5:
            return "moderate"
        else:
            return "weak"
    
    def _generate_market_summary(self, analyses: List[AnalysisResult]) -> Dict[str, Any]:
        """Generate executive market summary"""
        if not analyses:
            return {}
        
        # Latest policy stance
        latest_analysis = analyses[0] if analyses else None
        
        # Key market themes
        all_implications = []
        for analysis in analyses:
            all_implications.extend(analysis.market_implications)
        
        implication_counts = Counter(all_implications)
        
        # Policy trajectory assessment
        sentiment_scores = [analysis.sentiment_score for analysis in analyses]
        if len(sentiment_scores) >= 2:
            recent_trend = "tightening" if sentiment_scores[0] > sentiment_scores[-1] else "easing"
        else:
            recent_trend = "stable"
        
        # Risk assessment
        uncertainty_indicators = []
        for analysis in analyses:
            for theme in analysis.key_themes:
                if 'uncertainty' in theme.lower() or 'risk' in theme.lower():
                    uncertainty_indicators.append(theme)
        
        return {
            'current_policy_stance': latest_analysis.policy_stance if latest_analysis else "unknown",
            'current_sentiment_score': latest_analysis.sentiment_score if latest_analysis else 0,
            'policy_trajectory': recent_trend,
            'top_market_implications': implication_counts.most_common(5),
            'uncertainty_level': len(uncertainty_indicators),
            'key_risks': uncertainty_indicators[:3],
            'recommendation': self._generate_trading_recommendation(analyses)
        }
    
    def _generate_trading_recommendation(self, analyses: List[AnalysisResult]) -> str:
        """Generate high-level trading recommendation based on analysis"""
        if not analyses:
            return "Insufficient data for recommendation"
        
        latest = analyses[0]
        
        # Strong hawkish signals
        if latest.policy_stance == "hawkish" and latest.sentiment_score > 0.4:
            return "Consider positioning for higher rates: short duration, long USD, defensive equities"
        
        # Strong dovish signals
        elif latest.policy_stance == "dovish" and latest.sentiment_score < -0.4:
            return "Consider positioning for lower rates: long duration, growth equities, risk-on assets"
        
        # Mixed or neutral signals
        elif latest.policy_stance == "neutral" or abs(latest.sentiment_score) < 0.2:
            return "Mixed signals suggest range-bound markets: tactical positioning, volatility strategies"
        
        # Moderate signals
        else:
            return f"Moderate {latest.policy_stance} bias: gradual positioning, monitor for confirmation"

def export_to_json(transformed_data: Dict[str, Any], filename: str = None) -> str:
    """Export transformed data to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'fomc_analysis_{timestamp}.json'
    
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False, default=str)
    
    return filepath

def generate_executive_summary(transformed_data: Dict[str, Any]) -> str:
    """Generate executive summary of the analysis"""
    if not transformed_data:
        return "No data available for summary"
    
    summary_parts = []
    
    # Header
    summary_parts.append("=== FOMC MINUTES EXECUTIVE SUMMARY ===")
    summary_parts.append(f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}")
    summary_parts.append("")
    
    # Market summary
    if 'market_summary' in transformed_data:
        market = transformed_data['market_summary']
        summary_parts.append("MARKET OUTLOOK:")
        summary_parts.append(f"• Current Policy Stance: {market.get('current_policy_stance', 'Unknown').upper()}")
        summary_parts.append(f"• Sentiment Score: {market.get('current_sentiment_score', 0):.2f} (Range: -1.0 dovish to +1.0 hawkish)")
        summary_parts.append(f"• Policy Trajectory: {market.get('policy_trajectory', 'Unknown').upper()}")
        summary_parts.append("")
        
        if market.get('recommendation'):
            summary_parts.append("TRADING RECOMMENDATION:")
            summary_parts.append(f"• {market['recommendation']}")
            summary_parts.append("")
    
    # Comparative analysis
    if 'comparative_analysis' in transformed_data:
        comp = transformed_data['comparative_analysis']
        summary_parts.append("POLICY ANALYSIS:")
        
        if 'policy_stance_distribution' in comp:
            stances = comp['policy_stance_distribution']
            summary_parts.append("• Recent Policy Stance Distribution:")
            for stance, count in stances.items():
                summary_parts.append(f"  - {stance.title()}: {count} meetings")
        
        if 'most_common_themes' in comp:
            summary_parts.append("• Key Themes:")
            for theme, count in comp['most_common_themes'][:5]:
                summary_parts.append(f"  - {theme}: {count} mentions")
        
        summary_parts.append("")
    
    # Risk factors
    if 'market_summary' in transformed_data and 'key_risks' in transformed_data['market_summary']:
        risks = transformed_data['market_summary']['key_risks']
        if risks:
            summary_parts.append("KEY RISKS TO MONITOR:")
            for risk in risks:
                summary_parts.append(f"• {risk}")
            summary_parts.append("")
    
    # Data quality
    if 'analysis_metadata' in transformed_data:
        meta = transformed_data['analysis_metadata']
        summary_parts.append("DATA QUALITY:")
        summary_parts.append(f"• Documents Processed: {meta.get('documents_processed', 0)}")
        summary_parts.append(f"• Successful Analyses: {meta.get('successful_analyses', 0)}")
        summary_parts.append("")
    
    summary_parts.append("=== END SUMMARY ===")
    
    return "\n".join(summary_parts)

def main():
    """Test the transformation pipeline with sample data"""
    # Simple test data structure
    class TestDoc:
        def __init__(self, content, meeting_date=None):
            self.content = content
            self.meeting_date = meeting_date
            self.release_date = None
            self.document_type = "minutes"
    
    # Create sample documents
    sample_docs = [
        TestDoc(
            content="""
            The Federal Open Market Committee decided to raise the federal funds rate target range 
            by 25 basis points to 5.25-5.50 percent. Participants noted that inflation remains elevated 
            and the labor market continues to be tight. Several participants emphasized the need for 
            a more restrictive monetary policy stance to bring inflation back to the Committee's 
            2 percent objective. The Committee voted unanimously in favor of this decision.
            Economic growth has been robust, with strong consumer spending and business investment.
            Jerome H. Powell noted upside risks to inflation from strong wage growth.
            The unemployment rate remained low while GDP growth exceeded expectations.
            """,
            meeting_date="January 28-29, 2025"
        ),
        TestDoc(
            content="""
            The Federal Open Market Committee decided to maintain the federal funds rate target range 
            at 5.25-5.50 percent. Participants observed signs of slowing economic growth and 
            moderating inflation pressures. Several members expressed concerns about downside risks 
            to the economic outlook given recent banking sector stress and tightening credit conditions.
            The Committee noted the importance of being data dependent in future policy decisions.
            Two members dissented, preferring a 25 basis point reduction. Employment conditions 
            showed signs of softening with rising unemployment claims and slower job growth.
            Financial conditions have tightened considerably over the intermeeting period.
            """,
            meeting_date="March 18-19, 2025"
        )
    ]
    
    # Process with transformer
    transformer = FEDDataTransformer()
    results = transformer.process_documents(sample_docs)
    
    # Generate summary
    summary = generate_executive_summary(results)
    
    print("=== FED DATA TRANSFORMER TEST ===")
    print(summary)
    
    # Export results
    output_file = export_to_json(results)
    print(f"\n✅ Analysis exported to: {output_file}")
    print("\nTransformation pipeline test completed successfully!")
    
    return results

if __name__ == "__main__":
    main()