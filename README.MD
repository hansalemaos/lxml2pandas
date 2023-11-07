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
df = subprocess_parsing(
    htmldata,
    chunks=1,
    processes=5,
    fake_header=True,
    print_stdout=True,
    print_stderr=True,
)
#
firstlist = df.loc[
    df.aa_attr_values.str.contains("ovm-FixtureList ovm-Competition_Fixtures", na=False)
]
for k, i in firstlist.iterrows():
    df2 = df.loc[df.aa_element_id.isin(i.aa_all_children)]
    df3 = df2.loc[
        df2.aa_attr_values.isin(
            ["ovm-FixtureDetailsTwoWay_TeamName", "ovm-ParticipantOddsOnly_Odds"]
        )
    ]
    df4 = [df3.iloc[x : x + 5] for x in range(0, len(df3), 5)]
    for dframe in df4:
        if len(dframe) == 5:
            dfc = dframe["aa_attr_values"].value_counts()
            try:
                if (
                    dfc.loc["ovm-FixtureDetailsTwoWay_TeamName"] == 2
                    and dfc.loc["ovm-ParticipantOddsOnly_Odds"] == 3
                ):
                    print(dframe)
            except Exception:
                pass
chi = df.loc[df.aa_attr_values == "events-list__grid__event"].aa_all_children
for c in chi:
    (
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
    ).ds_color_print_all()

```