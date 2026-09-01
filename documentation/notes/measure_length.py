#!/usr/bin/env python3
"""Measure manuscript length against the BMC page-projection fit."""
import re, sys, subprocess
from pathlib import Path

TEX = Path(sys.argv[1] if len(sys.argv) > 1 else 'main.tex')
src = TEX.read_text().split('\n')

def strip_comment(l):
    out = []
    for i, c in enumerate(l):
        if c == '%' and (i == 0 or l[i-1] != '\\'):
            break
        out.append(c)
    return ''.join(out)

lines = [strip_comment(l) for l in src]
n = len(lines)

# region: \maketitle .. \bmhead{Supplementary  (= main body)
start = next(i for i, l in enumerate(lines) if '\\maketitle' in l)
end   = next(i for i, l in enumerate(lines) if '\\bmhead{Supplementary' in l)
app   = next((i for i, l in enumerate(lines) if '\\begin{appendices}' in l), n)

FLOAT = r'(figure\*?|table\*?|sidewaystable\*?)'
keep = [True] * n
floats = []
i = 0
while i < n:
    m = re.search(r'\\begin\{' + FLOAT + r'\}', lines[i])
    if m:
        base, depth, j = m.group(1), 1, i + 1
        while j < n:
            if re.search(r'\\begin\{' + re.escape(base) + r'\}', lines[j]): depth += 1
            if re.search(r'\\end\{' + re.escape(base) + r'\}', lines[j]):
                depth -= 1
                if depth == 0: break
            j += 1
        floats.append((i, base, '\n'.join(lines[i:j+1])))
        for k in range(i, min(j+1, n)): keep[k] = False
        i = j + 1
    else:
        i += 1

def words(s):
    s = re.sub(r'\\[a-zA-Z]+\*?', '', s)
    s = re.sub(r'[\{\}\[\]\$&\\~^_]', ' ', s)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-\.%]*", s))

body = sum(words(lines[k]) for k in range(start, end) if keep[k])

def cap_words(block):
    tot = 0
    for m in re.finditer(r'\\caption(\[[^\]]*\])?\{', block):
        k = m.end() - 1; depth = 0
        for idx in range(k, len(block)):
            if block[idx] == '{' and block[idx-1] != '\\': depth += 1
            elif block[idx] == '}' and block[idx-1] != '\\':
                depth -= 1
                if depth == 0:
                    tot += words(block[k+1:idx]); break
    return tot

main_floats = [f for f in floats if f[0] < app]
fig = sum(1 for f in main_floats if f[1].startswith('figure'))
tab = sum(1 for f in main_floats if 'table' in f[1])
caps = sum(cap_words(f[2]) for f in main_floats)

bib = Path(TEX.parent / 'bibfile.bib')
cited = set()
for m in re.finditer(r'\\cite\{([^}]*)\}', '\n'.join(lines)):
    cited |= {c.strip() for c in m.group(1).split(',')}
refs = len(cited)

pages = 3.73 + 0.686*(body/1000) + 0.673*fig + 0.343*tab + 0.334*(refs/10)

print(f"body words (excl floats) : {body:>6}   target  9,600-10,000")
print(f"main-text figures        : {fig:>6}   target  7")
print(f"main-text tables         : {tab:>6}   target  5")
print(f"main-text caption words  : {caps:>6}   target  <=650")
print(f"references cited         : {refs:>6}")
print(f"PROJECTED PUBLISHED PAGES: {pages:>6.1f}   target  <=20")
