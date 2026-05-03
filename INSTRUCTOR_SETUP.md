# Instructor Setup Guide

This guide is for the **course instructor only**. Complete these steps once before
creating the GitHub Classroom assignment link. Students do not need to read this file.

---

## Overview

This project uses [DVC (Data Version Control)](https://dvc.org) to distribute the
5 GB chest X-ray dataset. The images are stored in a shared Google Drive folder;
the repository only holds a small pointer file (`data/chest_xray.dvc`, ~200 bytes)
that tells DVC where to fetch the data from.

**Your one-time setup flow:**
1. Create a Google Drive shared folder
2. Update `.dvc/config` with its folder ID
3. Download the dataset and track it with DVC
4. Push data to Google Drive and pointer file to GitHub
5. Create the GitHub Classroom assignment from this repo

---

## Step 1 — Create a Google Drive Storage Folder

1. Go to [drive.google.com](https://drive.google.com) and sign in with your
   institutional Google account.
2. Create a new folder, e.g. `digicrome-m3-dataset`.
3. Set sharing to **"Anyone with the link can view"** — this lets students
   authenticate and pull without needing individual access grants.
4. Copy the **folder ID** from the URL:
   ```
   https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                           This part is the folder ID
   ```

---

## Step 2 — Update the DVC Remote Config

Open `.dvc/config` and replace the placeholder with your folder ID:

```ini
[core]
    remote = classroom-storage
    autostage = true
['remote "classroom-storage"']
    url = gdrive://1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74
```

Commit this change:

```bash
git add .dvc/config
git commit -m "Configure DVC remote storage"
```

---

## Step 3 — Install DVC with Google Drive Support

```bash
pip install "dvc[gdrive]>=3.0.0"
```

---

## Step 4 — Download the Dataset

```bash
# Install the Kaggle CLI if needed
pip install kaggle

# Place your kaggle.json API token at ~/.kaggle/kaggle.json
# then download and unzip
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

Verify the layout:

```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/     (1,341 images)
    │   └── PNEUMONIA/  (3,875 images)
    ├── val/
    │   ├── NORMAL/     (8 images)
    │   └── PNEUMONIA/  (8 images)
    └── test/
        ├── NORMAL/     (234 images)
        └── PNEUMONIA/  (390 images)
```

---

## Step 5 — Track the Dataset with DVC

```bash
dvc add data/chest_xray
```

This command:
- Computes an MD5 hash of every image in `data/chest_xray/`
- Creates the pointer file `data/chest_xray.dvc`
- Appends `/chest_xray` to `data/.gitignore` (so git never sees the raw images)

Commit the pointer file:

```bash
git add data/chest_xray.dvc data/.gitignore
git commit -m "Track chest X-ray dataset with DVC"
```

---

## Step 6 — Push the Data to Google Drive

```bash
dvc push
```

DVC will open a browser window asking you to authenticate with Google. Sign in
with the same account that owns the storage folder.

After authentication succeeds, DVC uploads the dataset to your Google Drive folder.
This is a one-time operation — subsequent pushes only upload changed files.

---

## Step 7 — Push Everything to GitHub

```bash
git push origin main
```

---

## Step 8 — Create the GitHub Classroom Assignment

1. Go to [classroom.github.com](https://classroom.github.com) and open your course.
2. Click **New Assignment → Individual Assignment**.
3. Under **Starter code repository**, select this repository.
4. Set the deadline and other options as needed.
5. Copy the invitation link and share it with students.

---

## What Students Will Do

When a student accepts the assignment, GitHub Classroom auto-creates a personal
copy of this repository. The student then:

1. Clones their repo.
2. Installs requirements (includes DVC).
3. Runs `dvc pull` — DVC downloads the dataset directly from your Google Drive folder.
4. Completes the TODOs and pushes code to their repo.

Students never need a Kaggle account. They never commit images.

---

## Re-using This Setup Next Year

If you need to update the dataset version:

```bash
# Replace/update the images in data/chest_xray/
dvc add data/chest_xray     # re-hashes and updates the .dvc file
dvc push                    # uploads changed files to Google Drive
git add data/chest_xray.dvc
git commit -m "Update dataset to v2"
git push
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `dvc push` authentication loop | Delete `~/.config/pydrive2fs/` and try again |
| Students report `ERROR: failed to pull` | Confirm the Drive folder sharing is set to "Anyone with the link can view" |
| `dvc push` is very slow | Normal for first push of 5 GB — subsequent pushes are incremental |
| Students on Windows get auth errors | They may need to run `dvc remote modify classroom-storage gdrive_use_service_account false` |
