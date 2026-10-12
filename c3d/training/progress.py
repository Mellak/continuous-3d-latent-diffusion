"""Append readable progress; roll back CSV rows beyond a resumed checkpoint."""
import csv
from pathlib import Path

def log(run,message):
    text=str(message)
    print(text,flush=True)
    path=Path(run)/'train.log';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:stream.write(text+'\n')

def trim_csv(path,key,maximum):
    path=Path(path)
    if not path.exists():return
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    if not fields or key not in fields:return
    retained=[row for row in rows if int(row[key])<=maximum]
    if len(rows)==len(retained):return
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(retained)
    temporary.replace(path)
