# arXiv submission

## 1. Preparation

### 1.1 Configure Paths

* Edit the file paths for `.tex` and image files in:

  ```
  a_copy-tex-figures.bash
  ```

### 1.2 Copy Files

* Run the script to copy all required `.tex` files and figures into the `arxiv/files` directory:

  ```bash
  bash a_copy-tex-figures.bash
  ```
* Ensure you also copy any required `.cls` files.

---

## 2. Update LaTeX File

### 2.1 Modify Figure Paths

In `files/main.tex`, replace:

```latex
\graphicspath{{../figures/*.png}}
```

with:

```latex
\includegraphics[width=\textwidth]{*.png}
```

### 2.2 Update Bibliography Path

Adjust the bibliography path as needed, for example:

```latex
\bibliography{../../references/references}
```

---

## 3. Initial Compilation

* Compile the document:

  ```bash
  cd files/
  bash ../b_pdflatex-bibtex.bash
  ```
* Verify that references are correctly rendered:

  ```bash
  evince main.pdf
  ```

---

## 4. Prepare arXiv Version

### 4.1 Inline Bibliography

In `main.tex`, replace:

```latex
%%\bibliography{../references/references}
```

with:

```latex
\input{main.bbl}  % Required for arXiv submission
```

### 4.2 Recompile

```bash
cd files/
bash ../c_pdflatex-pdflatex.bash
```

### 4.3 Verify Output

```bash
cd files/
evince main.pdf
```

* Ensure all figures and references render correctly.

### 4.4 Clean Project

```bash
cd files/
bash ../d_clean-tex-project.bash
```

---

## 5. Create Submission Archive

* Compress the project into a `.zip` file:

  ```bash
  cd ../
  bash e_zip_files.bash v00  # Version 00
  bash e_zip_files.bash v01  # Version 01
  ```

---

## 6. Final Output

Your submission package will be available as:

```
zip-files/arxiv-v00.zip
```

This file is ready for upload to arXiv. 🎉


## 2. Submission

1. **Start a New Submission**

   * Log in to arXiv: https://arxiv.org/login
   * Click **“Start New Submission”** and upload your prepared `.zip` file.

2. **Submission Agreement**
   Select and confirm the following:

   * I certify that the above information is correct
   * I have read and agree to the Instructions for Submission
   * I accept the arXiv Submission Terms and Agreement
   * I am submitting as an author of this article
   * License: **CC BY-SA (Creative Commons Attribution-ShareAlike)**
   * Archive and Subject Class: **Physics > Medical Physics**
   * Click **Continue**

3. **Upload Files**

   * Upload your file (e.g., `arxiv-v00.zip`)
   * Click **Continue → Process Files**
   * Wait until the status shows: **“Processing Status: Succeeded!”**
   * Click **Continue**

4. **Enter Metadata**

   * **Title**: Enter your paper title
   * **Authors**: Use full names (Firstname Lastname).

     * Do not use “et al.”
     * Separate authors with commas or “and”
   * **Abstract**: Paste your abstract
   * **Comments**: e.g., *N pages, N figures*
   * Click **Save and Continue**

5. **Select Categories**

   * Primary: **Medical Physics (physics.med-ph)**
   * Optional additional categories (adjust as needed):

     * Artificial Intelligence (cs.AI) — remove if not relevant
     * Hardware Architecture (cs.AR) — remove if not relevant
     * Machine Learning (cs.LG) — remove if not relevant
     * Image and Video Processing (eess.IV)

6. **Preview Submission**

   * Preview your compiled PDF carefully
   * Ensure formatting, figures, and references are correct
   * Refresh the page after previewing

7. **Submit**

   * Click **Submit**
   * Note: Processing may take several minutes

8. **Optional**

   * Maintain a submission record (e.g., `SubmissionLog.md`) for tracking versions and updates

---

## References

* arXiv metadata guidelines: https://arxiv.org/help/prep#title
