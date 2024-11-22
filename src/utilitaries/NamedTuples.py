from collections import namedtuple

# ---------------------------------------------------------------------------- #
#                            NAMEDTUPLES DEFINITION                            #
# ---------------------------------------------------------------------------- #

# Define namedtuple for downloaded CSV files
DownloadedData = namedtuple("DownloadedData", ["file_path", "content"])
DatasetResult = namedtuple("DatasetResult", ["train", "test"])
DownloadResult = namedtuple("DownloadResult", ["french_difficulty", "ljl", "sentences"])
PricingEstimation = namedtuple(
    "PricingEstimation", ["input_tokens", "output_tokens", "training_tokens", "total"]
)
