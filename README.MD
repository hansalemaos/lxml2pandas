# lxml to pandas for fast web scraping

## Tested against Windows / Python 3.11 / Anaconda

## pip install lxml2pandas

```python
from lxml2pandas import subprocess_parsing
from PrettyColorPrinter import add_printer

add_printer(1)

htmldata = [
    ("bet365", r"C:\Users\hansc\Downloads\bet365 - Apostas Desportivas Online.mhtml"),
    (
        "betano",
        r"C:\Users\hansc\Downloads\Brasil Brasileirão - Série A Apostas - Futebol Odds _ Betano.mhtml",
    ),
    ("sportingbet", r"C:\Users\hansc\Downloads\Apostas Futebol _ Sportingbet.mhtml"),
]
allframes = []
df = subprocess_parsing(
    htmldata,
    chunks=1,
    processes=5,
    fake_header=True,
    print_stdout=True,
    print_stderr=True,
)

for child in df.loc[df.aa_attr_values == "ovm-Fixture_Container"].aa_all_children:
    dfr = df.loc[
        df.aa_element_id.isin(child)
        & (
            (df.aa_attr_values == "ovm-FixtureDetailsTwoWay_TeamName")
            | (df.aa_attr_values == "ovm-ParticipantOddsOnly_Odds")
        )
    ]
    if len(dfr) == 5:
        print(dfr)

chi = df.loc[df.aa_attr_values == "events-list__grid__event"].aa_all_children
for c in chi:
    print(
        df.loc[
            (df.aa_element_id.isin(c))
            & (df.aa_doc_id == "betano")
            & (
                (
                    (df.aa_tag == "span")
                    & (df.aa_attr_values == "selections__selection__odd")
                )
                | (
                    (df.aa_tag == "span")
                    & (df.aa_attr_values.str.contains("participant-name", na=False))
                )
            )
        ]
    )


df = subprocess_parsing(
    htmldata,
    chunks=1,
    processes=5,
    fake_header=True,
    print_stdout=True,
    print_stderr=True,
)

# pre-filter
df0 = subprocess_parsing(
    htmldata,
    chunks=1,
    processes=5,
    fake_header=True,
    print_stdout=True,
    print_stderr=True,
    allowed_tags=("span", "div"),
)
df1 = subprocess_parsing(
    htmldata,
    chunks=1,
    processes=5,
    fake_header=True,
    print_stdout=True,
    print_stderr=True,
    allowed_tags=("span",),
)
df2 = subprocess_parsing(
    htmldata,
    chunks=1,
    processes=5,
    fake_header=True,
    print_stdout=True,
    print_stderr=True,
    allowed_tags=("span", "div"),
    allowed_attr=(
        "ovm-Fixture_Container",
        "ovm-FixtureDetailsTwoWay_TeamName",
        "ovm-ParticipantOddsOnly_Odds",
    ),
)

df3 = subprocess_parsing(
    htmldata,
    chunks=1,
    processes=5,
    fake_header=True,
    print_stdout=True,
    print_stderr=True,
    allowed_tags=(),
    allowed_attr=("ovm-ParticipantOddsOnly_Odds",),
    forbidden_tags=("p",),
)

df4 = subprocess_parsing(
    htmldata,
    chunks=1,
    processes=5,
    fake_header=True,
    print_stdout=True,
    print_stderr=True,
    allowed_tags=(),
    allowed_attr=(
        "ovm-Fixture_Container",
        "ovm-FixtureDetailsTwoWay_TeamName",
        "ovm-ParticipantOddsOnly_Odds",
        "events-list__grid__even",
        "selections__selection__odd",
        "events-list__grid__info__main__participants__participant-name tw-truncate",
    ),
    allowed_attr_keys=("class",),
)


df5 = subprocess_parsing(
    htmldata,
    chunks=1,
    processes=5,
    fake_header=True,
    print_stdout=True,
    print_stderr=True,
    allowed_tags=(),
    allowed_attr=("ovm-Fixture_Container",),
    allowed_attr_keys=("class",),
)

# parse a webpage:
df = subprocess_parsing(
    [("python", "https://www.python.org/")],
    chunks=1,
    processes=5,
    fake_header=True,
    print_stdout=True,
    print_stderr=True,
    allowed_tags=(),
    allowed_attr=(),
    allowed_attr_keys=(),
)



# Generate a column with css selectors 

from lxml2pandas import subprocess_parsing,pd_add_generate_css_selector
pd_add_generate_css_selector()
htmldata = [
    ("bet365", r"C:\Users\hansc\Downloads\bet365 - Apostas Desportivas Online2.mhtml"),
]
df = subprocess_parsing(
    htmldata,
    chunks=1,
    processes=5,
    fake_header=True,
    print_stdout=False,
    print_stderr=True,
)
df = df.s_generate_css_selector()

```