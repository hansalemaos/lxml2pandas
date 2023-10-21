# lxml to pandas for fast web scraping

## Tested against Windows / Python 3.11 / Anaconda

## pip install lxml2pandas

```python
from lxml2pandas import subprocess_parsing

df = subprocess_parsing(
    tmpfiles,
    chunks=1,
    processes=4,
    print_stdout=True,
    print_stderr=True,
    print_function_exceptions=True,
    children_and_parents=True,
    allowed_tags=('span',),
    filter_function=lambda x: 'odd' in (gh:=str(x.aa_attr_values).lower()) or 'team' in gh,
)

df = subprocess_parsing(
    tmpfiles,
    chunks=1,
    processes=4,
    print_stdout=True,
    print_stderr=True,
    print_function_exceptions=True,
    children_and_parents=False,
    allowed_tags=('span',),
    filter_function=lambda x: 'odd' in (gh := str(x.aa_attr_values).lower()) or 'team' in gh,
    # lambda x:'yt' in str(x.aa_attr_values).lower()  
)

df = subprocess_parsing(
    tmpfiles,
    chunks=1,
    processes=4,
    print_stdout=True,
    print_stderr=True,
    children_and_parents=True,
    allowed_tags=(),
    forbidden_tags=['div','body','html'],
    filter_function=lambda x:'Odd' in str(x.aa_html),

)
df = subprocess_parsing(
    tmpfiles,
    chunks=1,
    processes=4,
    print_stdout=True,
    print_stderr=True,
    print_function_exceptions=True,
    children_and_parents=False,
    allowed_tags=('div',),
    filter_function= lambda x:'team' in str(x.aa_attr_values).lower()
    # lambda x:'yt' in str(x.aa_attr_values).lower()   #  lambda x:x.aa_tag=='div' or str(x.aa_attr_keys) in ['class','href'] or 'mais' in str(x.aa_text)
)
df = subprocess_parsing(
    parsedata,
    chunks=1,
    processes=4,
    print_stdout=True,
    print_stderr=True,
    print_function_exceptions=True,
    children_and_parents=True,
    allowed_tags=(),
    forbidden_tags=('html','body'),
    filter_function= lambda x:'t' in str(x.aa_attr_values).lower()
)
```