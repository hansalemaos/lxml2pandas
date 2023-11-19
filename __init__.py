import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from email.parser import BytesParser
from email import policy
from functools import cache
import requests
from fake_headers import Headers
from flatten_everything import flatten_everything
from lxml import etree
from a_pandas_ex_apply_ignore_exceptions import pd_add_apply_ignore_exceptions
import pandas as pd
import numpy as np
from lxml import html
from multiprocnomain import start_multiprocessing
from pandas.core.frame import DataFrame


pd_add_apply_ignore_exceptions()

iswindows = "win" in platform.platform().lower()
if iswindows:
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    creationflags = subprocess.CREATE_NO_WINDOW
    invisibledict = {
        "startupinfo": startupinfo,
        "creationflags": creationflags,
        "start_new_session": True,
    }
else:
    invisibledict = {}


def get_fake_header():
    header = Headers(headers=False).generate()
    agent = header["User-Agent"]

    headers = {
        "User-Agent": f"{agent}",
    }

    return headers


def _add_css_selector(df):
    aa_all_css = (
        df.groupby(["aa_html", "aa_doc_id", "aa_multipart_counter", "aa_multipart"])
        .apply(
            lambda h: [
                str(h.aa_tag.to_list()[0])
                + "".join(
                    sorted(
                        set(
                            [
                                f'[{r[0]}="{r[1]}"]'
                                for r in (
                                    zip(
                                        h.aa_attr_keys.to_list(),
                                        h.aa_attr_values.to_list(),
                                    )
                                )
                                if r[0] and r[1]
                            ]
                        ),
                        key=len,
                    )
                ),
                h.index.to_list(),
            ]
        )
        .reset_index(drop=True)
        .apply(pd.Series)
    )
    return pd.concat([df, aa_all_css.explode(1).set_index(1)], axis=1)


def pd_add_generate_css_selector():
    DataFrame.s_generate_css_selector = _add_css_selector


def get_html_src(htmlcode, fake_header=True):
    if isinstance(htmlcode, str):
        if os.path.exists(htmlcode):
            if os.path.isfile(htmlcode):
                with open(htmlcode, mode="rb") as f:
                    htmlcode = f.read()
        elif re.search(r"^.{1,10}://", str(htmlcode)) is not None:
            if not fake_header:
                htmlcode = requests.get(htmlcode).content
            else:
                htmlcode = requests.get(htmlcode, headers=get_fake_header()).content
        else:
            htmlcode = htmlcode.encode("utf-8", "backslashreplace")
    return htmlcode


def multidata(multipart_messages, fake_header=True, musthave=()):
    allco = []
    totalno = 0
    for name, multipart_message in multipart_messages:
        multipart_message = get_html_src(multipart_message, fake_header=fake_header)
        message = BytesParser(policy=policy.default).parsebytes(multipart_message)
        con = 0

        for part in message.walk():
            try:
                content = part.get_payload(decode=True)
                if musthave:
                    for mus in musthave:
                        if mus in content:
                            break
                    else:
                        continue
                if content:
                    allco.append([con, content, name, totalno])
                    totalno = totalno + 1
                    con = con + 1

            except Exception as fe:
                sys.stderr.write(f"{fe}\n")
                sys.stderr.flush()
    return allco


def _parse_html(data):
    exec(f"from collections import defaultdict", globals())
    exec(
        f"from a_pandas_ex_apply_ignore_exceptions import pd_add_apply_ignore_exceptions",
        globals(),
    )
    exec(f"import pandas as pd", globals())
    exec(f"import numpy as np", globals())
    exec(f"from lxml import html, etree", globals())
    exec(f"from functools import cache", globals())

    # ,
    pd_add_apply_ignore_exceptions()

    idcounter = 0

    def fia(el, h):
        nonlocal idcounter
        elhash = hash(el)
        if elhash not in newids:
            newids[elhash] = int(idcounter)
            idcounter += 1
        if elhash not in allitems:
            allitems[elhash] = el
        if not hasattr(el, "getchildren"):
            method = el.iter
        else:
            method = el.getchildren
        for j in method():
            fia(j, h + (elhash,))
            allparents[hash(j)].update(h)
            allparentschildren[elhash].add(hash(j))

    allitems = {}
    allparents = defaultdict(set)
    allparentschildren = defaultdict(set)

    newids = {}
    tree = html.fromstring(data[1])
    fia(tree, ())
    allparents = {k: frozenset(v) for k, v in allparents.items()}
    allparentschildren = {k: frozenset(v) for k, v in allparentschildren.items()}
    df = pd.DataFrame(pd.Series(allitems))
    df["aa_element_id"] = df.index.__array__().copy()
    df["aa_parents"] = df.index.map(allparents)
    allchildren = defaultdict(set)
    for k, v in allparents.items():
        for vv in v:
            allchildren[vv].add(k)
    allchildren2 = {}
    for k in allchildren:
        allchildren2[k] = frozenset(allchildren[k])
    df["aa_all_children"] = df.index.map(allchildren2)
    df["aa_direct_children"] = df.index.map(allparentschildren)
    df["aa_tag"] = df[0].ds_apply_ignore(pd.NA, lambda q: q.tag)

    def parse_text(h):
        try:
            tco = "\n".join(([str(x.text_content()) for x in h])).strip()
            if not tco:
                return etree.tostring(
                    h, method="text", encoding="unicode", with_tail=True
                ).strip()
            return tco
        except Exception:
            try:
                return etree.tostring(
                    h, method="text", encoding="unicode", with_tail=True
                ).strip()
            except Exception:
                return ""

    df["aa_text"] = df[0].ds_apply_ignore(pd.NA, lambda q: parse_text(q))
    df["aa_items"] = df[0].ds_apply_ignore(((None, None),), lambda q: tuple(q.items()))
    df["aa_tail"] = df[0].ds_apply_ignore(pd.NA, lambda q: q.tail.strip())
    df["aa_html"] = df[0].ds_apply_ignore(
        pd.NA,
        lambda q: etree.tostring(
            q, method="html", encoding="unicode", pretty_print=False
        ).strip(),
    )
    df["aa_doc_id"] = data[2]
    df["aa_multipart_id"] = data[0]
    df["aa_multipart_counter"] = data[3]
    df = df.explode("aa_items").reset_index(drop=True)
    df["aa_attr_keys"] = df["aa_items"].ds_apply_ignore(
        pd.NA, lambda j: j[0] if not isinstance(j[0], str) else j[0].strip()
    )
    df["aa_attr_values"] = df["aa_items"].ds_apply_ignore(
        pd.NA, lambda j: j[1] if not isinstance(j[1], str) else j[1].strip()
    )
    df.drop(columns="aa_items", inplace=True)
    df = df.loc[
        ~(
            df[0].apply(lambda q: "_ElementTree" in str(type(q)))
            | (df.aa_tag == "html")
            | (df.aa_tag == "body")
        )
    ].reset_index(drop=True)

    lookupdict = {v: k for k, v in df.aa_element_id.to_dict().items()}
    df.aa_element_id = df.index.__array__().copy()

    @cache
    def get_vals(x):
        alli = []
        try:
            for y in x:
                try:
                    alli.append(lookupdict[y])
                except Exception:
                    continue
            return tuple(sorted(alli))
        except Exception:
            return ()

    df.aa_parents = df.aa_parents.apply(get_vals)
    df.aa_direct_children = df.aa_direct_children.apply(get_vals)
    df.aa_all_children = df.aa_all_children.apply(get_vals)
    df.loc[df.aa_tag.apply(callable), "aa_tag"] = "HTML_COMMENT"
    df = df[
        [
            "aa_multipart_id",
            "aa_element_id",
            "aa_tag",
            "aa_text",
            "aa_attr_keys",
            "aa_attr_values",
            "aa_parents",
            "aa_direct_children",
            "aa_all_children",
            "aa_html",
            "aa_tail",
            "aa_doc_id",
            "aa_multipart_counter",
        ]
    ].fillna("")
    unique_aa_tag = np.unique(df["aa_tag"].__array__())
    unique_aa_text = np.unique(df["aa_text"].__array__())
    unique_aa_attr_keys = np.unique(df["aa_attr_keys"].__array__())
    unique_aa_attr_values = np.unique(df["aa_attr_values"].__array__())
    unique_aa_html = np.unique(df["aa_html"].__array__())
    unique_aa_tail = np.unique(df["aa_tail"].__array__())
    unique_aa_doc_id = np.unique(df["aa_doc_id"].__array__())
    unique_aa_parents = set(df["aa_parents"])
    unique_aa_all_children = set(df["aa_all_children"])
    unique_aa_direct_children = set(df["aa_direct_children"])

    return [
        df.astype(
            {
                "aa_multipart_id": np.uint32,
                "aa_element_id": np.uint32,
                "aa_multipart_counter": np.uint32,
            }
        ),
        [
            unique_aa_tag,
            unique_aa_text,
            unique_aa_attr_keys,
            unique_aa_attr_values,
            unique_aa_html,
            unique_aa_tail,
            unique_aa_doc_id,
            unique_aa_parents,
            unique_aa_all_children,
            unique_aa_direct_children,
        ],
    ]


def _parse_html_filter(data, allowedtags, forbiddentags, allowedattr, allowedattrkeys):
    exec(f"from collections import defaultdict", globals())
    exec(
        f"from a_pandas_ex_apply_ignore_exceptions import pd_add_apply_ignore_exceptions",
        globals(),
    )
    exec(f"import pandas as pd", globals())
    exec(f"import numpy as np", globals())
    exec(f"from lxml import html, etree", globals())
    exec(f"from functools import cache", globals())
    pd_add_apply_ignore_exceptions()

    idcounter = 0

    def fia(el, h):
        nonlocal idcounter
        elhash = hash(el)
        if elhash not in newids:
            newids[elhash] = int(idcounter)
            idcounter += 1
        if elhash not in allitems:
            allitems[elhash] = el
        if not hasattr(el, "getchildren"):
            method = el.iter
        else:
            method = el.getchildren
        for j in method():
            fia(j, h + (elhash,))
            allparents[hash(j)].update(h)
            allparentschildren[elhash].add(hash(j))

    allitems = {}
    allparents = defaultdict(set)
    allparentschildren = defaultdict(set)

    newids = {}
    tree = html.fromstring(data[1])
    fia(tree, ())
    allparents = {k: frozenset(v) for k, v in allparents.items()}
    allparentschildren = {k: frozenset(v) for k, v in allparentschildren.items()}
    df = pd.DataFrame(pd.Series(allitems))
    df["aa_tag"] = df[0].ds_apply_ignore(pd.NA, lambda q: q.tag)
    if allowedtags:
        df = df.loc[(df.aa_tag.isin(allowedtags)) | df.aa_tag.isna()]
    if forbiddentags:
        df = df.loc[(~df.aa_tag.isin(forbiddentags)) | df.aa_tag.isna()]
    df["aa_items"] = df[0].ds_apply_ignore(((None, None),), lambda q: tuple(q.items()))

    def get_str_v(x):
        try:
            for xx in x:
                if xx[1] in allowedattr:
                    return True
        except Exception:
            return False
        return False

    def get_str_k(x):
        try:
            for xx in x:
                if xx[0] in allowedattrkeys:
                    return True
        except Exception:
            return False
        return False

    if allowedattr:
        df = df.loc[df["aa_items"].apply(get_str_v) | df.aa_tag.isna()]

    if allowedattrkeys:
        if allowedattr:
            df = df.loc[df["aa_items"].apply(get_str_k) | df.aa_tag.isna()]
    df = df.explode("aa_items")  # .reset_index(drop=True)
    df["aa_attr_keys"] = df["aa_items"].ds_apply_ignore(
        pd.NA, lambda j: j[0] if not isinstance(j[0], str) else j[0].strip()
    )
    df["aa_attr_values"] = df["aa_items"].ds_apply_ignore(
        pd.NA, lambda j: j[1] if not isinstance(j[1], str) else j[1].strip()
    )
    df["aa_element_id"] = df.index.__array__().copy()
    df["aa_parents"] = df.index.map(allparents)
    allchildren = defaultdict(set)
    for k, v in allparents.items():
        for vv in v:
            allchildren[vv].add(k)
    allchildren2 = {}
    for k in allchildren:
        allchildren2[k] = frozenset(allchildren[k])
    df["aa_all_children"] = df.index.map(allchildren2)
    df["aa_direct_children"] = df.index.map(allparentschildren)

    def parse_text(h):
        try:
            tco = "\n".join(([str(x.text_content()) for x in h])).strip()
            if not tco:
                return etree.tostring(
                    h, method="text", encoding="unicode", with_tail=True
                ).strip()
            return tco
        except Exception:
            try:
                return etree.tostring(
                    h, method="text", encoding="unicode", with_tail=True
                ).strip()
            except Exception:
                return ""

    df["aa_text"] = df[0].ds_apply_ignore(pd.NA, lambda q: parse_text(q))
    df["aa_tail"] = df[0].ds_apply_ignore(pd.NA, lambda q: q.tail.strip())
    df["aa_html"] = df[0].ds_apply_ignore(
        pd.NA,
        lambda q: etree.tostring(
            q, method="html", encoding="unicode", pretty_print=False
        ).strip(),
    )
    df["aa_doc_id"] = data[2]
    df["aa_multipart_id"] = data[0]
    df["aa_multipart_counter"] = data[3]

    df = df.loc[
        ~(
            df[0].apply(lambda q: "_ElementTree" in str(type(q)))
            | (df.aa_tag == "html")
            | (df.aa_tag == "body")
        )
    ].reset_index(drop=True)

    lookupdict = {v: k for k, v in df.aa_element_id.to_dict().items()}
    df.aa_element_id = df.index.__array__().copy()

    @cache
    def get_vals(x):
        alli = []
        try:
            for y in x:
                try:
                    alli.append(lookupdict.get(y, None))
                except Exception:
                    continue
            return tuple(sorted([j for j in alli if j is not None]))
        except Exception:
            return ()

    df.aa_parents = df.aa_parents.apply(get_vals)
    df.aa_direct_children = df.aa_direct_children.apply(get_vals)
    df.aa_all_children = df.aa_all_children.apply(get_vals)
    df.loc[df.aa_tag.apply(callable), "aa_tag"] = "HTML_COMMENT"
    df = df[
        [
            "aa_multipart_id",
            "aa_element_id",
            "aa_tag",
            "aa_text",
            "aa_attr_keys",
            "aa_attr_values",
            "aa_parents",
            "aa_direct_children",
            "aa_all_children",
            "aa_html",
            "aa_tail",
            "aa_doc_id",
            "aa_multipart_counter",
        ]
    ].fillna("")
    unique_aa_tag = np.unique(df["aa_tag"].__array__())
    unique_aa_text = np.unique(df["aa_text"].__array__())
    unique_aa_attr_keys = np.unique(df["aa_attr_keys"].__array__())
    unique_aa_attr_values = np.unique(df["aa_attr_values"].__array__())
    unique_aa_html = np.unique(df["aa_html"].__array__())
    unique_aa_tail = np.unique(df["aa_tail"].__array__())
    unique_aa_doc_id = np.unique(df["aa_doc_id"].__array__())
    unique_aa_parents = set(df["aa_parents"])
    unique_aa_all_children = set(df["aa_all_children"])
    unique_aa_direct_children = set(df["aa_direct_children"])

    return [
        df.astype(
            {
                "aa_multipart_id": np.uint32,
                "aa_element_id": np.uint32,
                "aa_multipart_counter": np.uint32,
            }
        ),
        [
            unique_aa_tag,
            unique_aa_text,
            unique_aa_attr_keys,
            unique_aa_attr_values,
            unique_aa_html,
            unique_aa_tail,
            unique_aa_doc_id,
            unique_aa_parents,
            unique_aa_all_children,
            unique_aa_direct_children,
        ],
    ]


def subprocess_parsing(
    htmldata,
    chunks=1,
    processes=5,
    fake_header=True,
    print_stdout=False,
    print_stderr=True,
    allowed_tags=(),
    forbidden_tags=(),
    allowed_attr=(),
    allowed_attr_keys=(),
):
    if any(
        (
            th := [
                allowed_tags,
                allowed_attr,
                forbidden_tags,
                allowed_attr_keys,
            ]
        )
    ):
        try:
            musthave = (tgh.encode("utf-8") for tgh in flatten_everything(th))
        except Exception:
            musthave = ()

        dada = [
            {
                "data": u,
                "allowedtags": allowed_tags,
                "forbiddentags": forbidden_tags,
                "allowedattr": allowed_attr,
                "allowedattrkeys": allowed_attr_keys,
            }
            for u in multidata(
                multipart_messages=htmldata, fake_header=fake_header, musthave=musthave
            )
        ]
        alldataready = start_multiprocessing(
            fu=_parse_html_filter,
            it=dada,
            processes=processes,
            chunks=chunks,
            print_stdout=print_stdout,
            print_stderr=print_stderr,
        )
        df = pd.concat([alldataready[q][0] for q in alldataready], ignore_index=True)
        alldataready = list(alldataready.values())

    else:
        dada = [
            {"data": u}
            for u in multidata(multipart_messages=htmldata, fake_header=fake_header)
        ]
        alldataready = start_multiprocessing(
            fu=_parse_html,
            it=dada,
            processes=processes,
            chunks=chunks,
            print_stdout=print_stdout,
            print_stderr=print_stderr,
        )
        df = pd.concat(
            [alldataready[q][0].assign(aa_multipart=q) for q in alldataready],
            ignore_index=True,
        )
        alldataready = list(alldataready.values())

    try:
        df["aa_tag"] = pd.Series(
            pd.Categorical(
                df["aa_tag"],
                categories=np.unique(np.hstack([o[1][0] for o in alldataready])),
            )
        )
    except Exception as e:
        sys.stderr.write(f"{e}\n\n")
        sys.stderr.flush()

    try:
        df["aa_text"] = pd.Series(
            pd.Categorical(
                df["aa_text"],
                categories=np.unique(np.hstack([o[1][1] for o in alldataready])),
            )
        )
    except Exception as e:
        sys.stderr.write(f"{e}\n\n")
        sys.stderr.flush()

    try:
        df["aa_attr_keys"] = pd.Series(
            pd.Categorical(
                df["aa_attr_keys"],
                categories=np.unique(np.hstack([o[1][2] for o in alldataready])),
            )
        )
    except Exception as e:
        sys.stderr.write(f"{e}\n\n")
        sys.stderr.flush()

    try:
        df["aa_attr_values"] = pd.Series(
            pd.Categorical(
                df["aa_attr_values"],
                categories=np.unique(np.hstack([o[1][3] for o in alldataready])),
            )
        )
    except Exception as e:
        sys.stderr.write(f"{e}\n\n")
        sys.stderr.flush()

    try:
        df["aa_html"] = pd.Series(
            pd.Categorical(
                df["aa_html"],
                categories=np.unique(np.hstack([o[1][4] for o in alldataready])),
            )
        )
    except Exception as e:
        sys.stderr.write(f"{e}\n\n")
        sys.stderr.flush()

    try:
        df["aa_tail"] = pd.Series(
            pd.Categorical(
                df["aa_tail"],
                categories=np.unique(np.hstack([o[1][5] for o in alldataready])),
            )
        )
    except Exception as e:
        sys.stderr.write(f"{e}\n\n")
        sys.stderr.flush()

    try:
        df["aa_doc_id"] = pd.Series(
            pd.Categorical(
                df["aa_doc_id"],
                categories=np.unique(np.hstack([o[1][6] for o in alldataready])),
            )
        )
    except Exception as e:
        sys.stderr.write(f"{e}\n\n")
        sys.stderr.flush()
    try:
        df["aa_parents"] = pd.Series(
            pd.Categorical(
                df["aa_parents"],
                categories=set.union(*[o[1][7] for o in alldataready]),
            )
        )
    except Exception as e:
        sys.stderr.write(f"{e}\n\n")
        sys.stderr.flush()

    try:
        df["aa_all_children"] = pd.Series(
            pd.Categorical(
                df["aa_all_children"],
                categories=set.union(*[o[1][8] for o in alldataready]),
            )
        )
    except Exception as e:
        sys.stderr.write(f"{e}\n\n")
        sys.stderr.flush()

    try:
        df["aa_direct_children"] = pd.Series(
            pd.Categorical(
                df["aa_direct_children"],
                categories=set.union(*[o[1][9] for o in alldataready]),
            )
        )
    except Exception as e:
        sys.stderr.write(f"{e}\n\n")
        sys.stderr.flush()

    return df
