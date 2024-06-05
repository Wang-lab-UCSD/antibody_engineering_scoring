"""Contains tools for retrieving the key datasets used for benchmarking."""
import os
import shutil
import zipfile
import wget
import subprocess

from ..constants import data_retrieval_constants as DRC



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


def retrieve_mason_dataset(project_dir):
    """Retrieves the Mason et al. dataset."""
    current_dir = os.getcwd()
    os.chdir(project_dir)

    os.chdir("extracted_data")
    if "mason" not in os.listdir():
        os.mkdir("mason")
    os.chdir("..")

    # We download the whole git repo which is not ideal because we really just want the two
    # files, but there isn't a...great...alternative to this -- (there are some not-great
    # alternatives). To cleanup, we remove all the unnecessary files once we're done.
    res = subprocess.Popen(["git", "clone", "--depth=1", "https://github.com/dahjan/DMS_opt"])
    output, err = res.communicate()

    if err:
        raise RuntimeError("Failed to download git repo DMS_opt. Could not process Mason dataset.")

    os.chdir(os.path.join("DMS_opt", "data"))
    shutil.copy("mHER_H3_AgNeg.csv", os.path.join(project_dir, "extracted_data",
        "mason", "mHER_H3_AgNeg.csv"))
    shutil.copy("mHER_H3_AgPos.csv", os.path.join(project_dir, "extracted_data",
        "mason", "mHER_H3_AgPos.csv"))

    os.chdir(project_dir)

    shutil.rmtree("DMS_opt")
    os.chdir(current_dir)


def retrieve_cognano_dataset(project_dir):
    """Retrieves the Cognano dataset."""
    current_dir = os.getcwd()
    os.chdir(os.path.join(project_dir, "extracted_data"))
    if "cognano" not in os.listdir():
        os.mkdir("cognano")

    os.chdir("cognano")
    train_filename = wget.download("https://huggingface.co/datasets/COGNANO/AVIDa-SARS-CoV-2/resolve/main/train.csv?download=true")
    test_filename = wget.download("https://huggingface.co/datasets/COGNANO/AVIDa-SARS-CoV-2/resolve/main/test.csv?download=true")
    os.rename(train_filename, "train.csv")
    os.rename(test_filename, "test.csv")

    os.chdir(current_dir)


def retrieve_il6_dataset(project_dir):
    """Retrieves the IL-6 dataset."""
    current_dir = os.getcwd()
    os.chdir(os.path.join(project_dir, "extracted_data"))
    if "il6" not in os.listdir():
        os.mkdir("il6")

    os.chdir("il6")
    _ = wget.download(DRC.IL6_DATASET)
    os.chdir(current_dir)
