import os
import glob
from pypdf import PdfWriter

def pdf_compiler(pdf_path="pdfs", report_name="MCS-Surrogate.pdf", PPE_Subjects={}, mcs_scores={}):
    pdfs = os.listdir(pdf_path)

    pdfs = sorted(pdfs)


    for check_string in ["kinematics", "kinetics", "segmentation"]:
        for pdf_name in pdfs:
            if check_string in pdf_name:
                print(f"Removing:{pdf_name}")
                pdfs.remove(pdf_name)



    if ".ipynb_checkpoints" in pdfs:
        pdfs.remove(".ipynb_checkpoints")

    for check_string in ["all_subject", "mcs_subject", "ground_truth"]:
        for pdf_name in pdfs:
            if check_string in pdf_name:
                try: 
                    pdfs.remove(pdf_name)
                    pdfs.insert(0,pdf_name)
                except Exception as e:
                    print(f"{pdf_name} not found")
        
    PPE_Subjects2session = {v:k for k,v in PPE_Subjects.items()}

    # Remove individual plots for non mcs scores
    keep_indices = []
    for i in range(len(pdfs)):
        name = pdfs[i].split("_")[0]
        if name in PPE_Subjects2session: 
            # print(f"Checking:{name}")
            if PPE_Subjects2session[name] in mcs_scores and mcs_scores[PPE_Subjects2session[name]] > 1:
                # print(f"keep_indices:{name} MCS:{mcs_scores[PPE_Subjects2session[name]]}")
                keep_indices.append(i)
            else: 
                continue
        elif "subject" in pdfs[i]:
            keep_indices.append(i)
        

    pdfs = [pdfs[i] for i in keep_indices]

    merger = PdfWriter()

    for pdf in pdfs:
        merger.append(os.path.join(pdf_path,pdf))

    merger.write(report_name)
    merger.close()