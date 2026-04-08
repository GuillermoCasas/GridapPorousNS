import PyPDF2
with open("theory/Cocquet et al. - 2021 - Error analysis for the finite element approximatio.pdf", "rb") as f:
    reader = PyPDF2.PdfReader(f)
    text = ""
    for i in range(min(5, len(reader.pages))): # read first 5 pages
        text += reader.pages[i].extract_text()
    print(text)
