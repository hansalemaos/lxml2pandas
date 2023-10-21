# lxml to pandas for fast web scraping

## Tested against Windows / Python 3.11 / Anaconda

## pip install lxml2pandas

```python
from lxml2pandas import subprocess_parsing
htmldata = [
    r"C:\Users\hansc\Downloads\bet365 - Apostas Desportivas Online2.mhtml",
    r"C:\Users\hansc\Downloads\bet365 - Apostas Desportivas Online.mhtml",
    r"C:\Users\hansc\Downloads\Your Repositories.mhtml",
    'https://pandas.pydata.org/docs/reference/api/pandas.concat.html'
]

df=subprocess_parsing(
    htmldata, chunks=1, processes=5, print_stdout=True, print_stderr=True
)

```