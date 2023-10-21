import io
import multiprocessing
import operator
import pickle
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from email.parser import BytesParser
from functools import cache, partial
import numpy as np
import re as regex
import requests
from lxml import etree
from email import policy
from fake_headers import Headers
from a_pandas_ex_apply_ignore_exceptions import pd_add_apply_ignore_exceptions
import pandas as pd
import os

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


def multidata(htmls):
    childexplode = False
    parentexplode = False
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
                        ]
                    )
                    con = con + 1
                    hierachcounter3 += 1

            except Exception as fe:
                sys.stderr.write(f"{fe}\n")
        hierachcounter1 += 1
    return allco


def nested_dict():
    return defaultdict(nested_dict)


def parsehtml2df(multiparseddata, shar_list):
    try:
        _parsehtml2df(multiparseddata, shar_list)
    except Exception as fe:
        sys.stderr.write(f"{fe}\n")
        sys.stderr.flush()


def _parsehtml2df(multiparseddata, shar_list):
    parser = etree.HTMLParser()
    file = multiparseddata[1]
    tree = etree.parse(io.BytesIO(file), parser)
    ellis = {}
    hierachyset = set()
    uniquecounter = -2
    hashcodesonly = {}
    d = nested_dict()
    alldo = []
    childexplode = multiparseddata[7]
    parentexplode = multiparseddata[8]

    @cache
    def len_min_check(u):
        return len(u), min(u)

    @cache
    def sortedfaster(x):
        return tuple(
            sorted(
                set(
                    (tuple(b) for b in ([z for z in y if z in uniqhag] for y in x) if b)
                ),
                key=len_min_check,
            )
        )

    def convert_to_normal_dict_simple(di, keyx):
        if isinstance(di, defaultdict):
            di = {
                k: convert_to_normal_dict_simple(v, (keyx + (k,)))
                for k, v in di.items()
                if not paresedelements[k]["aa_parents"].add(
                    tuple((hashcodesonly.get(x, x) for x in keyx))
                )
            }
        alldo.append(keyx)
        return di

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
    paresedelements = {}
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
            htmlcode = etree.tostring(item, method="text", encoding="unicode")
        except Exception:
            htmlcode = pd.NA
        paresedelements[key] = {
            "aa_tag": tag,
            "aa_tail": tail,
            "aa_text": text,
            "aa_parents": set(),
            "aa_children": set(),
            "aa_uniquehash": item1["uniquehash"],
            "aa_allitems": allitems,
            "aa_html": htmlcode,
            "aa_comment": str(comment),
        }
    hierachy = list(hierachyset)
    hierachy.sort(key=len, reverse=True)

    for i in hierachy:
        try:
            it = iter(i)
            firstkey = next(it)
            value = d[firstkey]
            iti = tuple(it)
            for x in iti:
                try:
                    paresedelements[firstkey]["aa_children"].add(
                        (hashcodesonly.get(x, x))
                    )
                except Exception:
                    pass

            for ini, element in enumerate(iti):
                for x in iti[ini + 1 :]:
                    try:
                        paresedelements[element]["aa_children"].add(
                            (hashcodesonly.get(x, x))
                        )
                    except Exception:
                        pass

                value = operator.itemgetter(element)(value)
        except Exception:
            continue
    _ = convert_to_normal_dict_simple(d, ())
    for key in paresedelements:
        paresedelements[key]["aa_parents"] = tuple(
            sorted(paresedelements[key]["aa_parents"], key=len)[-1]
        )
        paresedelements[key]["aa_children"] = tuple(
            (paresedelements[key]["aa_children"])
        )
    df = pd.DataFrame.from_dict(paresedelements, orient="index").assign(
        aa_alldata=multiparseddata[3],
        aa_p0=multiparseddata[4],
        aa_p1=multiparseddata[5],
        aa_p2=multiparseddata[6],
        aa_p3=multiparseddata[0],
        aa_p4=multiparseddata[2],
    )
    df.dropna(subset="aa_tag", inplace=True)
    df = df.loc[(df.aa_tag != "body") & (df.aa_tag != "html")]
    uniqhag = df.aa_uniquehash.to_list()
    df["aa_index_part"] = np.arange(len(df))

    df.loc[:, "aa_parents"] = df.aa_parents.ds_apply_ignore(
        (), lambda x: sortedfaster((x,))[0]
    )
    df.loc[:, "aa_children"] = df.aa_children.ds_apply_ignore(
        (), lambda x: sortedfaster((x,))[0]
    )
    df = (
        df.explode("aa_allitems", ignore_index=True)
        .assign(
            aa_attr_keys=lambda j: j.aa_allitems.str[0],
            aa_attr_values=lambda j: j.aa_allitems.str[1],
        )
        .drop(columns="aa_allitems", inplace=False)
        .astype(
            {
                "aa_uniquehash": pd.Int32Dtype(),
                "aa_index_part": pd.Int32Dtype(),
                "aa_p0": pd.Int32Dtype(),
                "aa_p1": pd.Int32Dtype(),
                "aa_p2": pd.Int32Dtype(),
                "aa_p3": pd.Int32Dtype(),
                "aa_p4": pd.Int32Dtype(),
            }
        )
        .fillna(pd.NA, inplace=False)[
            [
                "aa_tag",
                "aa_attr_keys",
                "aa_attr_values",
                "aa_parents",
                "aa_children",
                "aa_text",
                "aa_html",
                "aa_comment",
                "aa_tail",
                "aa_uniquehash",
                "aa_index_part",
                "aa_p0",
                "aa_p1",
                "aa_p2",
                "aa_p3",
                "aa_p4",
                "aa_alldata",
            ]
        ]
    )
    if parentexplode:
        df = df.explode("aa_parents", ignore_index=True)
    if childexplode:
        df = df.explode("aa_children", ignore_index=True)
    shar_list.append(df)


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
    htmldata, chunks=2, processes=5, print_stdout=True, print_stderr=True
):
    fi0, remo0 = get_tmpfile(suffix=".xxtmpxx")

    dic = {}
    procx = get_procs(processes)
    dic["htmls"] = [
        q for q in multidata(htmldata) if q and isinstance(q, list) and len(q) == 9
    ]
    dic["chunks"] = chunks
    dic["procs"] = procx if procx else os.cpu_count()
    dic["save_path"] = fi0
    v = pickle.dumps(dic, protocol=pickle.HIGHEST_PROTOCOL)
    fi, remo = get_tmpfile(suffix=".xxtmpxx")
    with open(fi, mode="wb") as f:
        f.write(v)
    p = subprocess.run(
        [sys.executable, __file__, fi],
        capture_output=True,
        **invisibledict
    )
    if print_stdout:
        print(p.stdout.decode("utf-8", "backslashreplace"))
    if print_stderr:
        sys.stderr.write(f"{p.stderr.decode('utf-8','backslashreplace')}\n\n")
    df = pd.read_pickle(fi0)
    #
    try:
        os.remove(fi0)
    except Exception:
        pass
    try:
        os.remove(fi)
    except Exception:
        pass
    return df


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
                    _=pd.concat(
                        iter(shared_list), ignore_index=True, copy=False
                    ).to_pickle(save_path)
