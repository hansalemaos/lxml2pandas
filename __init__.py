import io
import multiprocessing
import dill
import pickle
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from email.parser import BytesParser
from functools import partial
import re as regex
import requests
from lxml import etree
from email import policy
from fake_headers import Headers
from a_pandas_ex_apply_ignore_exceptions import pd_add_apply_ignore_exceptions
import pandas as pd
import os
from functools import reduce
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


def get_html_src(htmlcode, fake_header=True):
    if isinstance(htmlcode, str):
        if os.path.exists(htmlcode):
            if os.path.isfile(htmlcode):
                with open(htmlcode, mode="rb") as f:
                    htmlcode = f.read()
        elif regex.search(r"^.{1,10}://", str(htmlcode)) is not None:
            if not fake_header:
                htmlcode = requests.get(htmlcode).content
            else:
                htmlcode = requests.get(htmlcode, headers=get_fake_header()).content
        else:
            htmlcode = htmlcode.encode("utf-8", "backslashreplace")
    return htmlcode


def multidata(
    htmls,
    childexplode=False,
    parentexplode=False,
    children_and_parents=False,
    allowed_tags=(),
    filter_function=None,
):
    allco = []
    hierachcounter1 = 1
    hierachcounter2 = 2
    hierachcounter3 = 3

    for iht, html in enumerate(htmls):
        multipart_message = get_html_src(html)
        message = BytesParser(policy=policy.default).parsebytes(multipart_message)
        con = 0
        hierachcounter2 += 1

        for part in message.walk():
            try:
                content = part.get_payload(decode=True)

                if content:
                    allco.append(
                        [
                            con,
                            content,
                            iht,
                            html,
                            hierachcounter1,
                            hierachcounter2,
                            hierachcounter3,
                            childexplode,
                            parentexplode,
                            children_and_parents,
                            allowed_tags,
                            filter_function,
                        ]
                    )
                    con = con + 1
                    hierachcounter3 += 1

            except Exception as fe:
                sys.stderr.write(f"{fe}\n")
        hierachcounter1 += 1
    return allco


def _parsehtmldf(file):
    parser = etree.HTMLParser()
    tree = etree.parse(io.BytesIO(file), parser)
    ellis = {}
    ellis_children = defaultdict(set)
    ellis_parents = defaultdict(set)

    ellis_children_hierachy = defaultdict(set)

    hierachyset = set()
    uniquecounter = 0
    hashcodesonly = {}

    def rec2(t, number):
        nonlocal uniquecounter
        if not hasattr(t, "getchildren"):
            method = t.iter
        else:
            method = t.getchildren
        hierachyset.add(tuple(number))
        hashcode2 = hash(t)
        if hashcode2 not in ellis:
            uniquecounter += 1
            ellis[hashcode2] = {"element": t, "uniquehash": uniquecounter}
            hashcodesonly[hashcode2] = uniquecounter

        for x in method():
            hashcode = hash(x)
            ellis_children[hashcode2].add(hashcode)
            ellis_parents[hashcode].add(hashcode2)
            ellis_children_hierachy[hashcode2].add(tuple([hashcode]))

            if hashcode not in ellis:
                uniquecounter += 1
                ellis[hashcode] = {"element": x, "uniquehash": uniquecounter}
                hashcodesonly[hashcode] = uniquecounter
            if hashcode not in number:
                rec2(x, number + [hashcode])

            else:
                rec2(x, number)

    hashroot = hash(tree)
    hashcodesonly[hashroot] = uniquecounter
    ellis[hashroot] = {"element": tree, "uniquehash": uniquecounter}
    rec2(tree, [hashroot])
    return (
        ellis_children_hierachy,
        hashroot,
        hashcodesonly,
        ellis,
        ellis_parents,
        ellis_children,
    )


def parsehtml2df(multiparseddata, shared_list):
    try:
        multiparseddataparsing(multiparseddata, shared_list)
    except Exception as fe:
        sys.stderr.write(f"{fe}")
        return


def multiparseddataparsing(multiparseddata, shared_list):
    file = get_html_src(multiparseddata[1])
    (
        ellis_children_hierachy,
        hashroot,
        hashcodesonly,
        ellis,
        ellis_parents,
        ellis_children,
    ) = _parsehtmldf(file)

    forbidden_tags = multiparseddata[7]
    print_function_exceptions = multiparseddata[8]
    children_and_parents = multiparseddata[9]
    allowed_tags = multiparseddata[10]

    filter_function = dill.loads(multiparseddata[11])

    tagdict = {}
    commentdict = {}
    taildict = {}
    textdict = {}
    itemsdict = {}
    htmlcodedict = {}
    for key, item1 in ellis.items():
        item = item1["element"]
        try:
            tag = item.tag
        except Exception:
            tag = pd.NA

        try:
            if callable(tag):
                comment = str(tag())
                tag = "HTML_COMMENT"

            else:
                comment = pd.NA
                tag = str(tag) if not pd.isna(tag) else tag

        except Exception:
            comment = pd.NA

        try:
            tail = item.tail.strip()
        except Exception:
            tail = pd.NA
        try:
            text = item.text.strip()
        except Exception:
            text = pd.NA
        try:
            allitems = tuple(item.items())
            if not allitems:
                allitems = ((pd.NA, pd.NA),)
        except Exception:
            allitems = ((pd.NA, pd.NA),)
        try:
            htmlcode = etree.tostring(item, method="text", encoding="unicode").strip()
        except Exception:
            htmlcode = pd.NA
        tagdict[key] = tag
        commentdict[key] = str(comment)
        taildict[key] = tail
        textdict[key] = text
        itemsdict[key] = allitems
        htmlcodedict[key] = htmlcode

    dfchildrenhierachy = (
        pd.DataFrame(ellis_children_hierachy.items())
        .set_index(0)
        .rename(columns={1: "aa_children_hierachy"})
    )
    dfhashcodesonly = (
        pd.DataFrame(hashcodesonly.items())
        .set_index(0)
        .rename(columns={1: "aa_hashcodes_only"})
    )
    dfellis_parents = (
        pd.DataFrame(ellis_parents.items())
        .set_index(0)
        .rename(columns={1: "aa_all_parents"})
    )

    dfellis_children = (
        pd.DataFrame(ellis_children.items())
        .set_index(0)
        .rename(columns={1: "aa_all_children"})
    )
    dftagdict = pd.DataFrame(tagdict.items()).set_index(0).rename(columns={1: "aa_tag"})

    dfcommentdict = (
        pd.DataFrame(commentdict.items()).set_index(0).rename(columns={1: "aa_comment"})
    )
    dftaildict = (
        pd.DataFrame(taildict.items()).set_index(0).rename(columns={1: "aa_tail"})
    )
    dftextdict = (
        pd.DataFrame(textdict.items()).set_index(0).rename(columns={1: "aa_text"})
    )
    dfitemsdict = (
        pd.DataFrame(itemsdict.items())
        .set_index(0)
        .rename(columns={1: "aa_allitems"})
        .explode("aa_allitems")
    )  #
    dfitemsdict["aa_attr_keys"] = dfitemsdict["aa_allitems"].ds_apply_ignore(
        pd.NA, lambda j: j[0]
    )
    dfitemsdict["aa_attr_values"] = dfitemsdict["aa_allitems"].ds_apply_ignore(
        pd.NA, lambda j: j[1]
    )
    dfitemsdict.drop(columns=["aa_allitems"], inplace=True)
    dfhtmlcodedict = (
        pd.DataFrame(htmlcodedict.items()).set_index(0).rename(columns={1: "aa_html"})
    )

    dfellis_children["aa_all_children"] = dfellis_children[
        "aa_all_children"
    ].ds_apply_ignore((), tuple)
    dfellis_parents["aa_all_parents"] = dfellis_parents[
        "aa_all_parents"
    ].ds_apply_ignore((), tuple)
    dfchildrenhierachy["aa_children_hierachy"] = dfchildrenhierachy[
        "aa_children_hierachy"
    ].ds_apply_ignore((), tuple)
    if children_and_parents:
        dfs = [
            dftagdict,
            dfitemsdict,
            dftextdict,
            dfellis_parents,
            dfellis_children,
            dfcommentdict,
            dftaildict,
            dfhashcodesonly,
            dfhtmlcodedict,
            dfchildrenhierachy,
        ]
    else:
        dfs = [
            dftagdict,
            dfitemsdict,
            dftextdict,
            # dfellis_parents,
            # dfellis_children,
            dfcommentdict,
            dftaildict,
            dfhashcodesonly,
            dfhtmlcodedict,
            # dfchildrenhierachy,
        ]
    df = reduce(
        lambda a, b: a.merge(b, right_index=True, left_index=True, how="outer"), dfs
    )
    if allowed_tags:
        df = df.loc[(df.aa_tag.isin(allowed_tags))].reset_index(drop=True)
    if forbidden_tags:
        df = df.loc[~(df.aa_tag.isin(forbidden_tags))].reset_index(drop=True)

    if filter_function:
        df = df.loc[
            df.apply(
                lambda x: apply_pandas_function(
                    filter_function, x, printexception=print_function_exceptions
                ),
                axis=1,
            )
        ]

    shared_list.append(
        df.assign(
            aa_alldata=multiparseddata[3],
            aa_p0=multiparseddata[4],
            aa_p1=multiparseddata[5],
            aa_p2=multiparseddata[6],
            aa_p3=multiparseddata[0],
            aa_p4=multiparseddata[2],
        )
    )


def get_procs(processes):
    if not processes:
        processes = os.cpu_count() - 1
    return processes


def get_tmpfile(suffix=".txt"):
    tfp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    filename = tfp.name
    filename = os.path.normpath(filename)
    tfp.close()
    return filename, partial(os.remove, tfp.name)


def subprocess_parsing(
    htmldata,
    chunks=2,
    processes=5,
    print_function_exceptions=True,
    print_stdout=True,
    print_stderr=True,
    children_and_parents=True,
    allowed_tags=(),
    filter_function=None,
    forbidden_tags=("html", "body"),
):
    if not isinstance(htmldata, (list, tuple)):
        htmldata = [htmldata]
    htmldatafiles = []
    deletefilefunctions = []
    for da in htmldata:
        if isinstance(da, bytes):
            _fi, _remove = get_tmpfile(suffix=".mhtml")
            with open(_fi, mode="wb") as f:
                f.write(da)
                htmldatafiles.append(_fi)
                deletefilefunctions.append(_remove)
        elif isinstance(da, str):
            if ">" in da or "<" in da:
                _fi, _remove = get_tmpfile(suffix=".mhtml")
                with open(_fi, mode="w", encoding="utf-8") as f:
                    f.write(da)
                htmldatafiles.append(_fi)
                deletefilefunctions.append(_remove)
        else:
            htmldatafiles.append(da)
    htmldata = htmldatafiles
    fi0, remo0 = get_tmpfile(suffix=".xxtmpxx")
    deletefilefunctions.append(remo0)
    dic = {}
    filterfunctionpickled = dill.dumps(
        filter_function, protocol=pickle.HIGHEST_PROTOCOL
    )
    procx = get_procs(processes)
    dic["htmls"] = [
        q
        for q in multidata(
            htmldata,
            childexplode=forbidden_tags,
            parentexplode=print_function_exceptions,
            children_and_parents=children_and_parents,
            allowed_tags=allowed_tags,
            filter_function=filterfunctionpickled,
        )
        if q and isinstance(q, list) and len(q) == 12
    ]
    dic["chunks"] = chunks
    dic["procs"] = procx if procx else os.cpu_count()
    dic["save_path"] = fi0
    v = pickle.dumps(dic, protocol=pickle.HIGHEST_PROTOCOL)
    fi, remo = get_tmpfile(suffix=".xxtmpxx")
    deletefilefunctions.append(remo)
    with open(fi, mode="wb") as f:
        f.write(v)
    p = subprocess.run(
        [sys.executable, __file__, fi], capture_output=True, **invisibledict
    )
    if print_stdout:
        print(p.stdout.decode("utf-8", "backslashreplace"))
    if print_stderr:
        sys.stderr.write(f"{p.stderr.decode('utf-8', 'backslashreplace')}\n\n")
    df = pd.read_pickle(fi0)
    for fun in deletefilefunctions:
        try:
            fun()
        except Exception:
            pass
    return df


def get_fake_header():
    header = Headers(headers=False).generate()
    agent = header["User-Agent"]

    headers = {
        "User-Agent": f"{agent}",
    }

    return headers


def apply_pandas_function(fu, x, printexception=True):
    try:
        return fu(x)
    except Exception as fe:
        if printexception:
            sys.stderr.write(f"{fe}\n")
        return False


def get_html_src(htmlcode, fake_header=True):
    if isinstance(htmlcode, str):
        if os.path.exists(htmlcode):
            if os.path.isfile(htmlcode):
                with open(htmlcode, mode="rb") as f:
                    htmlcode = f.read()
        elif regex.search(r"^.{1,10}://", str(htmlcode)) is not None:
            if not fake_header:
                htmlcode = requests.get(htmlcode).content
            else:
                htmlcode = requests.get(htmlcode, headers=get_fake_header()).content
        else:
            htmlcode = htmlcode.encode("utf-8", "backslashreplace")
    return htmlcode


if __name__ == "__main__":
    if len(sys.argv) == 2:
        fi = sys.argv[1]
        if fi.endswith(".xxtmpxx"):
            if os.path.exists(fi):
                with open(fi, mode="rb") as f:
                    data = f.read()
                dic = pickle.loads(data)
                htmls = dic["htmls"]
                chunks = dic["chunks"]
                processes = dic["procs"]
                save_path = dic["save_path"]
                with multiprocessing.Manager() as manager:
                    shared_list = manager.list()
                    with multiprocessing.Pool(processes=processes) as pool:
                        pool.starmap(
                            parsehtml2df,
                            ((value, shared_list) for value in htmls),
                            chunksize=chunks,
                        )
                        _ = pd.concat(
                            iter(shared_list), ignore_index=True, copy=False
                        ).to_pickle(save_path)
