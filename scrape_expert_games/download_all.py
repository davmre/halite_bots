import requests
import os

base_dir = "hlt_files"

with open("fnames.txt", "r") as f:
    lines = f.readlines()
entries = [line.strip().split(" ") for line in lines]

for username, fname in entries:
    url = "http://s3.amazonaws.com/halitereplaybucket/%s" % fname
    r = requests.get(url)

    target_dir = os.path.join(base_dir, username)
    try:
        os.makedirs(target_dir)
    except Exception as e:
        if e.errno != 17:
            print e

    saved_path = os.path.join(target_dir, fname)
    with open(saved_path, "w") as f:
        f.write(r.content)

    print "saved", saved_path


