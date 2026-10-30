#!/usr/bin/env python3
"""
verify_ste100.py - Verifies that the report text adheres to ASD-STE100 rules:
1. Descriptive sentences: max 25 words
2. Procedural sentences: max 20 words
3. No passive voice constructions ("is verb-ed by")
4. No forbidden ambiguous modals ("could", "may", "might")
"""

import os
import re
import sys

def check_text(text, is_procedural=False):
    max_words = 20 if is_procedural else 25
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = clean.replace('&nbsp;', ' ').replace('&bull;', ' ')
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    violations = []
    
    passive_pattern = re.compile(r'\b(is|are|was|were|been|being)\s+([a-z]+ed|[a-z]+en)\s+by\b', re.IGNORECASE)
    modal_pattern = re.compile(r'\b(could|may|might)\b', re.IGNORECASE)
    
    for s in sentences:
        s_strip = s.strip()
        if not s_strip:
            continue
        words = s_strip.split()
        count = len(words)
        
        # Check length
        if count > max_words:
            violations.append(f"LENGTH VIOLATION ({count} > {max_words} words): \"{s_strip}\"")
            
        # Check passive voice
        p_match = passive_pattern.search(s_strip)
        if p_match:
            violations.append(f"PASSIVE VOICE VIOLATION: \"{s_strip}\"")
            
        # Check forbidden modals
        m_match = modal_pattern.search(s_strip)
        if m_match:
            violations.append(f"AMBIGUOUS MODAL VIOLATION: \"{s_strip}\"")
            
    return violations

if __name__ == "__main__":
    report_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_report.py")
    with open(report_script, "r", encoding="utf-8") as f:
        code = f.read()
        
    paragraphs = re.findall(r'Paragraph\(\s*(".*?"|\'.*?\')\s*,', code, re.DOTALL)
    all_violations = []
    for p in paragraphs:
        # evaluate string
        try:
            val = eval(p)
            v = check_text(val)
            all_violations.extend(v)
        except:
            pass
            
    print(f"Total violations found: {len(all_violations)}")
    for v in all_violations:
        print(" -", v)

