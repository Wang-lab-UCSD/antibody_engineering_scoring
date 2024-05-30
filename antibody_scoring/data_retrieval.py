"""Contains tools for retrieving the key datasets used for benchmarking."""
import os
import zipfile
import wget

from .constants import data_retrieval_constants as DRC



def retrieve_engelhart_dataset(project_dir):
    """Retrieves the Engelhart et al. dataset."""
    os.chdir(project_dir)
    filename = wget.download(DRC.ENGELHART_DATASET)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(project_dir, "extracted_data", "engelhart"))

    os.chdir(os.path.join(project_dir, "extracted_data", "engelhart", DRC.LOCAL_ENGELHART_PATH[0]))

    with zipfile.ZipFile("MITLL_AAlphaBio_Ab_Binding_dataset.csv.zip", "r") as zip_ref:
        zip_ref.extractall(DRC.LOCAL_ENGELHART_PATH[1])

    os.remove("MITLL_AAlphaBio_Ab_Binding_dataset.csv.zip")

    os.chdir(project_dir)
    os.remove(filename)


def retrieve_desautels_dataset(project_dir):
    """Retrieves the Desautels et al. dataset."""
    os.chdir(project_dir)
    filename = wget.download(DRC.DESAUTELS_DATASET)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(project_dir, "extracted_data", "desautels"))

    os.chdir(os.path.join(project_dir, "extracted_data", "desautels", DRC.LOCAL_DESAUTELS_PATH[0]))

    for s in os.listdir():
        if s.endswith(".pdb") or s.endswith(".pdf"):
            os.remove(s)

    os.chdir(project_dir)
    os.remove(filename)

